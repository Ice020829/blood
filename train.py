import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# 导入优化的数据处理类
from cached_data_preprocess import GlucoseDatasetSliceOptimized
from model import VGG16Regressor1D
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import time


def calculate_mard(true, pred):
    """计算MARD (Mean Absolute Relative Difference)"""
    true = np.asarray(true).squeeze()
    pred = np.asarray(pred).squeeze()
    mask = true > 0
    return np.mean(np.abs(true[mask] - pred[mask]) / true[mask]) * 100


class MARDEarlyStopping:
    """基于MARD的早停机制"""
    def __init__(self, patience=10, min_delta=0.1, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.mard_history = []
        self.best_mard = float('inf')
        self.early_stop = False
        
    def __call__(self, current_mard, model):
        self.mard_history.append(current_mard)
        
        if current_mard < self.best_mard:
            self.best_mard = current_mard
            self.save_checkpoint(model)
            
        if len(self.mard_history) >= self.patience:
            start_mard = self.mard_history[-self.patience]
            end_mard = self.mard_history[-1]
            improvement = start_mard - end_mard
            
            if improvement < self.min_delta:
                print(f"MARD improvement over last {self.patience} epochs ({improvement:.4f}) is less than threshold ({self.min_delta})")
                self.early_stop = True
            
    def save_checkpoint(self, model):
        """保存模型"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        # 如果模型是DataParallel包装的，我们需要保存其module
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
            
        print(f"Saved model with best MARD: {self.best_mard:.4f}%")


def train_vgg16regressor1d():
    """训练并验证VGG16Regressor1D模型（使用DataParallel多GPU和优化的数据加载）"""
    start_time = time.time()
    
    # 配置数据路径
    data_dir = '/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明'
    xlsx_paths = [
        "/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明/无创血糖测试记录表(1).xlsx",
        "/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明/无创血糖测试记录表(1).xlsx"
    ]
    
    # 设置缓存目录
    cache_dir = '/home/pivot/zhenghongjie/裁剪信号_vgg/dataset_cache'
    
    print("Creating optimized dataset with caching...")
    # 创建优化的数据集
    dataset = GlucoseDatasetSliceOptimized(
        data_dir=data_dir,
        xlsx_paths=xlsx_paths,
        window_size=100,
        step_size=50,
        sampling_rate=100,
        train=True,
        cache_dir=cache_dir,
        use_cache=True,
        num_workers=8  # 使用8个进程并行处理数据
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 划分训练集、验证集和测试集
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # 数据加载器
    # 调整batch_size以适应多GPU
    batch_size = 256  # 增大批次大小以更好地利用多GPU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 使用VGG16Regressor1D模型
    model = VGG16Regressor1D(input_channels=2).to(device)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU并行训练")
        model = nn.DataParallel(model)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    
    # 对于多GPU训练，可以考虑稍微增大学习率
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # 初始化基于MARD的早停
    model_save_path = '/home/pivot/zhenghongjie/裁剪信号_vgg/best_model_vgg_multi_gpu.pth'
    early_stopping = MARDEarlyStopping(patience=15, min_delta=0.05, path=model_save_path)
    
    # 训练轮数
    num_epochs = 50
    
    # 存储训练过程中的指标
    train_losses = []
    val_losses = []
    val_mards = []
    
    print("开始训练 VGG16Regressor1D 模型 (使用DataParallel多GPU)...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * signals.size(0)
            
            # 打印批次进度
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss = epoch_loss / len(train_set)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_set)
        val_losses.append(val_loss)
        
        # 计算当前验证集的MARD
        val_preds = np.array(all_preds)
        val_labels = np.array(all_labels)
        current_mard = calculate_mard(val_labels, val_preds)
        val_mards.append(current_mard)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MARD: {current_mard:.4f}%, "
              f"LR: {current_lr:.6f}, "
              f"Time: {epoch_time:.2f}s")
        
        # 早停检查
        early_stopping(current_mard, model)
        if early_stopping.early_stop:
            print("Early stopping triggered due to minimal MARD improvement")
            break
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # 加载最佳模型进行评估
    if torch.cuda.device_count() > 1:
        # 创建非并行模型用于加载权重
        best_model = VGG16Regressor1D(input_channels=2).to(device)
        best_model.load_state_dict(torch.load(model_save_path))
    else:
        model.load_state_dict(torch.load(model_save_path))
        best_model = model
        
    best_model.eval()
    
    # 对验证集进行预测
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = best_model(signals)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    
    # 计算验证集评估指标
    val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
    val_r2 = r2_score(val_labels, val_preds)
    val_mard = calculate_mard(val_labels, val_preds)
    
    print(f"\n=== Validation Set Results ===")
    print(f"RMSE: {val_rmse:.2f}")
    print(f"R²: {val_r2:.4f}")
    print(f"MARD: {val_mard:.2f}%")
    
    # 对测试集进行预测和评估
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = best_model(signals)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    # 计算测试集评估指标
    test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    test_r2 = r2_score(test_labels, test_preds)
    test_mard = calculate_mard(test_labels, test_preds)
    
    print(f"\n=== Test Set Results ===")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"R²: {test_r2:.4f}")
    print(f"MARD: {test_mard:.2f}%")
    
    # 返回测试集的MARD
    return test_mard


if __name__ == '__main__':
    print("=== Training VGG16Regressor1D Model for Glucose Prediction ===")
    # 设置确定性随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    vgg_mard = train_vgg16regressor1d()
    
    print("\n=== Final Result ===")
    print(f"VGG16Regressor1D MARD: {vgg_mard:.2f}%")