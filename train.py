import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from data_preprocess import GlucoseDataset
from model import PytorchWaveletSVM, WaveletSVMPredictor
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import pickle


def calculate_mard(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    mask = true > 0  # 避免除零
    return np.mean(np.abs(true[mask] - pred[mask]) / true[mask]) * 100


def train_svm_model():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64  # SVM可以处理更大的批次
    
    # 设置保存路径
    save_dir = "/home/pivot/zhenghongjie/wavelet_svm"
    os.makedirs(save_dir, exist_ok=True)
    
    xlsx_paths = ["/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明/无创血糖测试记录表(1).xlsx",
                  "/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明/无创血糖测试记录表(1).xlsx"]
    
    # 数据集
    dataset = GlucoseDataset('/home/pivot/zhenghongjie/第二批 中山无创血糖训练样本及相关记录说明/', xlsx_paths)
    print(f"Dataset size: {len(dataset)}")
    
    # 添加检查：确保数据集不为空
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Please check your data loading process.")
    
    # 划分数据集为训练集(70%)、验证集(15%)和测试集(15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 确保每个部分至少有1个样本
    if train_size == 0 or val_size == 0 or test_size == 0:
        raise ValueError(f"Dataset too small to split: total={len(dataset)}, train={train_size}, val={val_size}, test={test_size}")
    
    # 使用两步划分方式，确保兼容性
    temp_dataset, test_set = random_split(dataset, [train_size + val_size, test_size])
    train_set, val_set = random_split(temp_dataset, [train_size, val_size])
    
    print(f"Split sizes - Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    
    # 验证数据集大小
    for name, dataset in [("Train", train_set), ("Validation", val_set), ("Test", test_set)]:
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty after splitting!")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    # 收集所有训练数据
    print("Collecting training data...")
    X_train_list, y_train_list, static_train_list = [], [], []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Loading training data")):
        # 检查批次大小
        if len(batch) == 0:
            print(f"Warning: Empty batch {batch_idx} encountered, skipping")
            continue
            
        # 解包数据
        if len(batch) == 4:
            seq, label, static_input, lengths = batch
        elif len(batch) == 3:
            seq, label, static_input = batch
        else:
            seq, label = batch
            static_input = None
        
        # 检查批次维度
        if seq.shape[0] == 0:
            print(f"Warning: Batch {batch_idx} has 0 samples, skipping")
            continue
            
        X_train_list.append(seq.numpy())
        y_train_list.append(label.numpy())
        if static_input is not None:
            static_train_list.append(static_input.numpy())
    
    # 检查是否收集到数据
    if not X_train_list:
        raise ValueError("No training data was collected! Check your DataLoader and GlucoseDataset implementation.")
    
    # 合并所有批次
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # 将y转换为1D数组，修复警告
    y_train = y_train.ravel()  # 将列向量转换为1D数组
    
    # 检查静态特征
    if static_train_list:
        static_train = np.concatenate(static_train_list, axis=0)
        # 检查静态特征形状
        if static_train.shape[0] == 0:
            print("Warning: Static features have 0 samples, setting to None")
            static_train = None
    else:
        static_train = None
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    if static_train is not None:
        print(f"Static features shape: {static_train.shape}")
    
    # 修改WaveletSVMPredictor类以处理空数据（这是假设的修复，实际需要修改model.py文件）
    class FixedWaveletSVMPredictor(WaveletSVMPredictor):
        def fit(self, X, y, static_features=None):
            # 检查输入数据是否为空
            if X.shape[0] == 0:
                raise ValueError("Cannot fit model with empty data (0 samples)")
                
            # 其余代码与原始类相同
            return super().fit(X, y, static_features)
    
    # 使用修复后的预测器类
    try:
        # 创建和训练SVM模型
        print("\nTraining SVM model...")
        svm_model = FixedWaveletSVMPredictor(
            wavelet='db4',
            level=3,
            kernel='rbf',
            use_grid_search=True  # 使用网格搜索找到最佳参数
        )
    except NameError:
        # 如果无法修改类，则使用原始类但添加验证
        print("\nTraining SVM model with additional validation...")
        if X_train.shape[0] == 0:
            raise ValueError("Cannot train model with empty data (0 samples)")
            
        svm_model = WaveletSVMPredictor(
            wavelet='db4',
            level=3,
            kernel='rbf',
            use_grid_search=True
        )
    
    # 训练模型前检查数据
    if X_train.shape[0] == 0:
        raise ValueError("Cannot train model with empty data (0 samples)")
    
    svm_model.fit(X_train, y_train, static_train)
    
    # 验证
    print("\nEvaluating on validation set...")
    X_val_list, y_val_list, static_val_list = [], [], []
    
    for batch in tqdm(val_loader, desc="Loading validation data"):
        # 解包数据
        if len(batch) == 4:
            seq, label, static_input, lengths = batch
        elif len(batch) == 3:
            seq, label, static_input = batch
        else:
            seq, label = batch
            static_input = None
        
        # 检查批次维度
        if seq.shape[0] == 0:
            print("Warning: Validation batch has 0 samples, skipping")
            continue
            
        X_val_list.append(seq.numpy())
        y_val_list.append(label.numpy())
        if static_input is not None:
            static_val_list.append(static_input.numpy())
    
    # 检查是否收集到验证数据
    if not X_val_list:
        print("Warning: No validation data was collected!")
        val_mard = float('nan')
    else:
        # 合并所有批次
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        # 将y转换为1D数组
        y_val = y_val.ravel()
        
        static_val = np.concatenate(static_val_list, axis=0) if static_val_list else None
        
        # 预测验证集
        y_val_pred = svm_model.predict(X_val, static_val)
        
        # 计算验证集MARD
        val_mard = calculate_mard(y_val, y_val_pred)
        print(f'Validation MARD: {val_mard:.2f}%')
    
    # 测试
    print("\nEvaluating on test set...")
    X_test_list, y_test_list, static_test_list = [], [], []
    
    for batch in tqdm(test_loader, desc="Loading test data"):
        # 解包数据
        if len(batch) == 4:
            seq, label, static_input, lengths = batch
        elif len(batch) == 3:
            seq, label, static_input = batch
        else:
            seq, label = batch
            static_input = None
        
        # 检查批次维度
        if seq.shape[0] == 0:
            print("Warning: Test batch has 0 samples, skipping")
            continue
            
        X_test_list.append(seq.numpy())
        y_test_list.append(label.numpy())
        if static_input is not None:
            static_test_list.append(static_input.numpy())
    
    # 检查是否收集到测试数据
    if not X_test_list:
        print("Warning: No test data was collected!")
        test_mard = float('nan')
    else:
        # 合并所有批次
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 将y转换为1D数组
        y_test = y_test.ravel()
        
        static_test = np.concatenate(static_test_list, axis=0) if static_test_list else None
        
        # 预测测试集
        y_test_pred = svm_model.predict(X_test, static_test)
        
        # 计算测试集MARD
        test_mard = calculate_mard(y_test, y_test_pred)
        print(f'Test MARD: {test_mard:.2f}%')
    
    # 保存模型 - 方法1：使用joblib
    model_path_joblib = os.path.join(save_dir, 'best_wavelet_svm_model.joblib')
    try:
        svm_model.save_model(model_path_joblib)
        print(f"Model saved to '{model_path_joblib}'")
    except Exception as e:
        print(f"Error saving model with joblib: {e}")
        
        # 备用方法：直接保存模型对象
        try:
            joblib.dump(svm_model, model_path_joblib)
            print(f"Model saved directly to '{model_path_joblib}'")
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
    
    # 保存模型 - 方法2：使用pickle
    model_path_pickle = os.path.join(save_dir, 'best_wavelet_svm_model.pkl')
    try:
        with open(model_path_pickle, 'wb') as f:
            pickle.dump(svm_model, f)
        print(f"Model saved with pickle to '{model_path_pickle}'")
    except Exception as e:
        print(f"Error saving model with pickle: {e}")
    
    # 保存PyTorch包装版本
    pytorch_model_path = os.path.join(save_dir, 'wavelet_svm_pytorch_model.pth')
    try:
        pytorch_model = PytorchWaveletSVM(
            input_dim=X_train.shape[2],
            static_dim=static_train.shape[1] if static_train is not None else 0
        )
        pytorch_model.svm_model = svm_model
        pytorch_model.is_fitted = True
        
        # 保存整个模型对象而不是state_dict
        torch.save(pytorch_model, pytorch_model_path)
        print(f"PyTorch model saved to '{pytorch_model_path}'")
    except Exception as e:
        print(f"Error saving PyTorch model: {e}")
        
        # 尝试只保存必要的属性
        try:
            model_data = {
                'input_dim': X_train.shape[2],
                'static_dim': static_train.shape[1] if static_train is not None else 0,
                'svm_model': svm_model
            }
            torch.save(model_data, pytorch_model_path.replace('.pth', '_data.pth'))
            print(f"Model data saved to '{pytorch_model_path.replace('.pth', '_data.pth')}'")
        except Exception as e2:
            print(f"Second attempt to save PyTorch model failed: {e2}")
    
    # 保存训练结果
    results_path = os.path.join(save_dir, 'training_results.pkl')
    try:
        results = {
            'val_mard': val_mard,
            'test_mard': test_mard,
            'val_predictions': y_val_pred if 'y_val_pred' in locals() else None,
            'val_true_values': y_val if 'y_val' in locals() else None,
            'test_predictions': y_test_pred if 'y_test_pred' in locals() else None,
            'test_true_values': y_test if 'y_test' in locals() else None,
            'best_params': svm_model.model.best_params_ if hasattr(svm_model, 'use_grid_search') and svm_model.use_grid_search else None
        }
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Training results saved to '{results_path}'")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # 列出保存目录中的所有文件
    print("\nFiles in save directory:")
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # 转换为KB
        print(f"- {file}: {file_size:.1f} KB")
    
    return val_mard, test_mard


def load_saved_model(model_path):
    """加载保存的模型"""
    print(f"Attempting to load model from: {model_path}")
    
    # 尝试不同的加载方法
    if model_path.endswith('.joblib'):
        try:
            model = joblib.load(model_path)
            print("Model loaded successfully with joblib")
            return model
        except Exception as e:
            print(f"Error loading with joblib: {e}")
    
    elif model_path.endswith('.pkl'):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully with pickle")
            return model
        except Exception as e:
            print(f"Error loading with pickle: {e}")
    
    elif model_path.endswith('.pth'):
        try:
            model = torch.load(model_path)
            print("Model loaded successfully with torch")
            return model
        except Exception as e:
            print(f"Error loading with torch: {e}")
    
    print("Failed to load model")
    return None


# 另外，需要修改WaveletSVMPredictor类中的StandardScaler使用部分
# 这部分代码应该在model.py文件中

"""
# 在model.py中找到WaveletSVMPredictor类的fit方法，添加数据验证：

def fit(self, X, y, static_features=None):
    # 确保输入数据不为空
    if X.shape[0] == 0:
        raise ValueError("Cannot fit model with empty data (0 samples)")
        
    # 提取小波特征
    X_features = self._extract_wavelet_features(X)
    
    # 处理静态特征
    if static_features is not None:
        # 检查静态特征是否为空
        if static_features.shape[0] == 0:
            print("Warning: Static features have 0 samples, ignoring static features")
        else:
            # 合并特征
            X_features = np.hstack((X_features, static_features))
    
    # 标准化特征之前检查
    if X_features.shape[0] == 0:
        raise ValueError("Cannot standardize empty features array")
        
    # 标准化特征
    if self.standardize:
        # 修复：仅当有样本时才使用StandardScaler
        if X_features.shape[0] > 0:
            self.scaler = StandardScaler()
            X_features = self.scaler.fit_transform(X_features)
    
    # 网格搜索或直接训练
    if self.use_grid_search:
        # 仅当有足够样本时使用网格搜索
        if X_features.shape[0] >= 5:  # 确保有足够样本进行交叉验证
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]}
            grid_search = GridSearchCV(
                SVR(kernel=self.kernel), 
                param_grid, 
                cv=min(5, X_features.shape[0]),  # 确保cv不超过样本数
                scoring='neg_mean_absolute_error'
            )
            grid_search.fit(X_features, y)
            self.model = grid_search
        else:
            print("Warning: Not enough samples for grid search, using default parameters")
            self.model = SVR(kernel=self.kernel)
            self.model.fit(X_features, y)
    else:
        self.model = SVR(kernel=self.kernel)
        self.model.fit(X_features, y)
    
    self.is_fitted = True
    return self
"""


if __name__ == '__main__':
    # 使用标准SVM训练
    val_mard, test_mard = train_svm_model()
    print(f"\nFinal results - Validation MARD: {val_mard:.2f}%, Test MARD: {test_mard:.2f}%")
    
    # 测试模型加载
    save_dir = "/home/pivot/zhenghongjie/wavelet_svm"  # 修正为一致的保存路径
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            if file.endswith(('.joblib', '.pkl', '.pth')):
                model_path = os.path.join(save_dir, file)
                loaded_model = load_saved_model(model_path)
                if loaded_model:
                    print(f"Successfully verified loading of {file}")