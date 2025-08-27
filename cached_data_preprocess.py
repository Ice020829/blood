import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import signal
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import pickle
import time
import hashlib


class GlucoseDatasetSliceOptimized(Dataset):
    def __init__(self, data_dir, xlsx_paths, window_size=100, step_size=50, sampling_rate=100, train=True, 
                 cache_dir=None, use_cache=True, num_workers=4):
        """
        Args:
            data_dir (str): 数据文件夹
            xlsx_paths (list): Excel文件列表
            window_size (int): 每个切片长度（帧数），默认100帧=1秒
            step_size (int): 切片步长（帧数），默认50帧=0.5秒滑动
            sampling_rate (int): 采样率
            train (bool): 是否训练模式（影响归一化）
            cache_dir (str): 缓存目录，如果为None，则使用data_dir下的cache子目录
            use_cache (bool): 是否使用缓存
            num_workers (int): 并行处理工作进程数
        """
        self.window_size = window_size
        self.step_size = step_size
        self.train = train
        self.data_info = pd.concat([pd.read_excel(p) for p in xlsx_paths], ignore_index=True)
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers
        
        # 设置缓存目录
        if cache_dir is None:
            self.cache_dir = os.path.join(data_dir, 'cache')
        else:
            self.cache_dir = cache_dir
            
        # 确保缓存目录存在
        if use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 生成缓存文件名（基于参数哈希）
        config_str = f"{data_dir}_{window_size}_{step_size}_{sampling_rate}_{train}"
        for path in xlsx_paths:
            config_str += f"_{os.path.basename(path)}"
        
        hash_obj = hashlib.md5(config_str.encode())
        self.cache_file = os.path.join(self.cache_dir, f"glucose_data_{hash_obj.hexdigest()}.pkl")
        
        # 保存所有小样本
        self.samples = []
        self.labels = []
        
        # 标准化器
        self.scaler = StandardScaler()
        
        # 尝试加载缓存，如果失败则重新处理数据
        if use_cache and os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            self._load_from_cache()
        else:
            print("Processing dataset from raw files...")
            start_time = time.time()
            # 加载数据
            self._load_and_slice_data()
            
            if train and len(self.samples) > 0:
                # 标准化处理
                flat_samples = np.concatenate([s.reshape(-1, 2) for s in self.samples], axis=0)
                self.scaler.fit(flat_samples)
                self.samples = [self.scaler.transform(s.reshape(-1, 2)).reshape(self.window_size, 2) for s in self.samples]
            
            # 转换为张量
            self.samples = [torch.FloatTensor(s) for s in self.samples]
            
            # 保存缓存
            if use_cache and len(self.samples) > 0:
                self._save_to_cache()
                
            print(f"Dataset processing completed in {time.time() - start_time:.2f} seconds")

    def _parse_hex(self, filepath):
        """读取红外/红光信号 - 优化版本"""
        ir, red = [], []
        
        with open(filepath, 'r') as f:
            # 一次性读取所有行，避免频繁IO
            lines = f.readlines()
            
        # 预分配内存
        n_lines = len(lines)
        ir = np.zeros(n_lines, dtype=np.int32)
        red = np.zeros(n_lines, dtype=np.int32)
        
        valid_count = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 17:
                continue
                
            try:
                # 直接使用16进制转换，避免从字节转换
                ir_bytes = [int(parts[6], 16), int(parts[7], 16), int(parts[8], 16), int(parts[9], 16)]
                red_bytes = [int(parts[10], 16), int(parts[11], 16), int(parts[12], 16), int(parts[13], 16)]
                
                ir_value = ir_bytes[0] + (ir_bytes[1] << 8) + (ir_bytes[2] << 16) + (ir_bytes[3] << 24)
                red_value = red_bytes[0] + (red_bytes[1] << 8) + (red_bytes[2] << 16) + (red_bytes[3] << 24)
                
                ir[valid_count] = ir_value
                red[valid_count] = red_value
                valid_count += 1
            except (ValueError, IndexError):
                # 处理可能的错误
                continue
                
        # 截取有效数据
        return ir[:valid_count], red[:valid_count]

    def _process_file(self, item):
        """处理单个文件并返回结果 - 用于并行处理"""
        idx, row = item
        filename = row['数据文件名']
        glucose_value = row['指尖血糖值']
        if pd.isna(glucose_value):
            return [], []
        
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return [], []
        
        # 解析数据
        ir, red = self._parse_hex(filepath)
        min_len = min(len(ir), len(red))
        
        if min_len == 0:
            return [], []
        
        # 预处理 - 滤波
        ir = signal.medfilt(ir[:min_len], kernel_size=5)
        red = signal.medfilt(red[:min_len], kernel_size=5)
        
        signals = np.stack([ir, red], axis=-1)  # [length, 2]
        
        # 按窗口滑动切片
        file_samples = []
        file_labels = []
        
        start = 0
        while start + self.window_size <= len(signals):
            slice_signal = signals[start:start+self.window_size, :]
            file_samples.append(slice_signal)
            file_labels.append(glucose_value)  # 每一小片用同样血糖标注
            start += self.step_size
            
        return file_samples, file_labels

    def _load_and_slice_data(self):
        """加载所有数据文件并切片 - 使用并行处理"""
        # 准备数据项列表
        items = [(idx, row) for idx, row in self.data_info.iterrows()]
        
        # 使用多进程并行处理
        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(self._process_file, items)
        else:
            # 单进程处理
            results = [self._process_file(item) for item in items]
        
        # 合并结果
        for file_samples, file_labels in results:
            self.samples.extend(file_samples)
            self.labels.extend(file_labels)
        
        print(f"Loaded {len(self.samples)} slices from {len(results)} files")

    def _save_to_cache(self):
        """保存处理好的数据到缓存文件"""
        print(f"Saving dataset to cache: {self.cache_file}")
        cache_data = {
            'samples': self.samples,
            'labels': self.labels,
            'scaler': self.scaler if self.train else None
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_from_cache(self):
        """从缓存文件加载处理好的数据"""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.samples = cache_data['samples']
        self.labels = cache_data['labels']
        if self.train and cache_data['scaler'] is not None:
            self.scaler = cache_data['scaler']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回：[2通道, 时间步, 1]，方便给 CNN
        sample = self.samples[idx].permute(1, 0).unsqueeze(-1)  # [2, window_size, 1]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label