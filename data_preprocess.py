import os
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class GlucoseDataset(Dataset):
    def __init__(self, data_dir, xlsx_paths, max_len=100, train=True):
        """
        Args:
            data_dir (str): Directory containing the infrared data files
            xlsx_path (str): Path to Excel file containing filenames and glucose values
            max_len (int): Maximum sequence length
            train (bool): Whether this is training data (affects standardization)
        """
        self.max_len = max_len
        self.train = train
        self.ir_weight = 0.7  # 940nm weight
        self.red_weight = 0.3  # 620nm weight
        
        # Load data mapping from Excel
        self.data_info1 = pd.read_excel(xlsx_paths[0])
        self.data_info2 = pd.read_excel(xlsx_paths[1])
        
        # Load and process data
        self.sequences, self.labels, self.lengths, self.static_input = self._load_data(data_dir)
        
        # Data standardization
        self.scaler = StandardScaler()
        if train:
            # Reshape for scaler: (n_samples, max_len*2) -> (n_samples*max_len, 2)
            original_shape = self.sequences.shape
            sequences_reshaped = self.sequences.reshape(-1, 2)
            self.scaler.fit(sequences_reshaped)
            self.sequences = self.scaler.transform(sequences_reshaped).reshape(original_shape)
        else:
            original_shape = self.sequences.shape
            sequences_reshaped = self.sequences.reshape(-1, 2)
            self.sequences = self.scaler.transform(sequences_reshaped).reshape(original_shape)
    
    def _parse_hex(self, filepath):
        """Parse infrared and red light signals from hex file"""
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        ir, red = [], []
        for line in lines:
            bytes_list = line.split()
            if len(bytes_list) != 17:
                continue
            
            # Parse and apply weights
            ir.append(int.from_bytes([int(b,16) for b in bytes_list[6:10]], 'little') * self.ir_weight)
            red.append(int.from_bytes([int(b,16) for b in bytes_list[10:14]], 'little') * self.red_weight)
        
        return np.array(ir), np.array(red)
    
    def _calculate_pulse_rate(self, red_signal, fs=100, min_hr=40, max_hr=180):
        
        b, a = signal.butter(4, [0.5, 5], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, red_signal)
        
        
        min_peak_distance = fs * 60 / max_hr
        peaks, _ = signal.find_peaks(filtered, 
                                distance=min_peak_distance,
                                prominence=np.std(filtered)*0.5)
        
        if len(peaks) < 2:
            return 0, []
        
        rr_intervals = np.diff(peaks) / fs  
        pulse_rate = 60 / np.mean(rr_intervals)
        
        return pulse_rate, peaks
    
    def _calculate_amplitude_diff_percentage(self, signal, window_size=100):
        diff_percent = []
        for i in range(len(signal) - window_size):
            window = signal[i:i+window_size]
            max_val = np.max(window)
            min_val = np.min(window)
            percent = (max_val - min_val) / ((max_val + min_val)/2 + 1e-10) * 100
            diff_percent.append(percent)
        return np.array(diff_percent)

    def _extract_features(self, ir, red):
        """Extract time-series features"""
        # Preprocessing
        ir = signal.medfilt(ir, kernel_size=5)
        red = signal.medfilt(red, kernel_size=3)
        
        # Truncate/pad sequence
        seq_len = min(len(ir), self.max_len)
        padded_seq = np.zeros((self.max_len, 2))
        padded_seq[:seq_len, 0] = ir[:seq_len]
        padded_seq[:seq_len, 1] = red[:seq_len]
        
        return padded_seq, seq_len
    
    def _load_data(self, data_dir):
        sequences = []
        lengths = []
        labels = []
        static_input = []
        
        for _, row in self.data_info1.iterrows():
            filename = row['数据文件名']  # Assuming column name is 'filename'
            glucose_value = row['指尖血糖值']  # Assuming column name is 'glucose'
            
            if pd.isna(glucose_value):
                continue
        # Parse signals
            try:
                ir, red = self._parse_hex(os.path.join(data_dir, filename))

                pr_red, red_peaks = self._calculate_pulse_rate(red, fs=100)
                static_input.append(pr_red)
                seq, seq_len = self._extract_features(ir, red)
                
                sequences.append(seq)
                lengths.append(seq_len)
                labels.append(float(glucose_value))

            except FileNotFoundError:
                print(f"Warning: File {filename} not found, skipping")
                continue

        for _, row in self.data_info2.iterrows():
            filename = row['数据文件名']  # Assuming column name is 'filename'
            glucose_value = row['指尖血糖值']  # Assuming column name is 'glucose'
            
            if pd.isna(glucose_value):
                continue
        # Parse signals
            try:
                ir, red = self._parse_hex(os.path.join(data_dir, filename))

                pr_red, red_peaks = self._calculate_pulse_rate(red, fs=100)
                static_input.append(pr_red)
                seq, seq_len = self._extract_features(ir, red)
                
                sequences.append(seq)
                lengths.append(seq_len)
                labels.append(float(glucose_value))
            except FileNotFoundError:
                print(f"Warning: File {filename} not found, skipping")
                continue
        
        return np.array(sequences), np.array(labels), np.array(lengths), np.array(static_input)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]]),
            torch.FloatTensor([self.static_input[idx]]),
            self.lengths[idx]
        )