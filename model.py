import torch
import torch.nn as nn
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib


class SimpleWaveletFeatureExtractor:
    def __init__(self, wavelet='db4', level=3):
        """
        简化版小波特征提取器，专注于最重要的特征
        :param wavelet: 小波基函数
        :param level: 小波分解层数
        """
        self.wavelet = wavelet
        self.level = level
    
    def extract_features(self, signal):
        """
        提取简化但有效的小波特征
        :param signal: 输入信号 [seq_len]
        :return: 特征向量
        """
        # 小波分解
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        features = []
        
        # 提取每个层级的关键统计特征
        for i, coeff in enumerate(coeffs):
            # 基本统计特征
            features.append(np.mean(coeff))
            features.append(np.std(coeff))
            features.append(np.sqrt(np.mean(coeff**2)))  # RMS
            
            # 能量
            features.append(np.sum(coeff**2))
            
            # 最大最小值
            features.append(np.max(np.abs(coeff)))
        
        # 添加重要的时域特征
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.max(signal) - np.min(signal),  # 范围
            np.mean(np.abs(np.diff(signal)))  # 平均变化率
        ])
        
        return np.array(features)


class WaveletSVMPredictor:
    """小波变换 + SVM 血糖预测器"""
    def __init__(
        self,
        wavelet='db4',
        level=3,
        kernel='rbf',
        use_grid_search=True
    ):
        self.wavelet = wavelet
        self.level = level
        self.kernel = kernel
        self.use_grid_search = use_grid_search
        
        # 特征提取器
        self.feature_extractor = SimpleWaveletFeatureExtractor(wavelet=wavelet, level=level)
        
        # 特征标准化
        self.scaler = StandardScaler()
        
        # SVM模型
        if use_grid_search:
            # 使用网格搜索优化超参数
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
            self.model = GridSearchCV(
                SVR(kernel=kernel),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        else:
            self.model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    
    def extract_features(self, X):
        """
        从时间序列数据中提取特征
        :param X: 输入数据 [n_samples, seq_len, input_dim]
        :return: 特征矩阵
        """
        n_samples, seq_len, input_dim = X.shape
        features_list = []
        
        for i in range(n_samples):
            sample_features = []
            for dim in range(input_dim):
                dim_features = self.feature_extractor.extract_features(X[i, :, dim])
                sample_features.extend(dim_features)
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def fit(self, X, y, static_features=None):
        """
        训练模型
        :param X: 输入数据 [n_samples, seq_len, input_dim]
        :param y: 目标值
        :param static_features: 静态特征 [n_samples, static_dim]
        """
        # 确保y是1D数组
        if len(y.shape) > 1:
            y = y.ravel()
        
        # 提取特征
        features = self.extract_features(X)
        
        # 合并静态特征
        if static_features is not None:
            features = np.concatenate([features, static_features], axis=1)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练SVM
        self.model.fit(features_scaled, y)
        
        if self.use_grid_search:
            print(f"Best parameters: {self.model.best_params_}")
            print(f"Best score: {-self.model.best_score_:.4f}")
    
    def predict(self, X, static_features=None):
        """
        预测
        :param X: 输入数据 [n_samples, seq_len, input_dim]
        :param static_features: 静态特征 [n_samples, static_dim]
        :return: 预测值
        """
        # 提取特征
        features = self.extract_features(X)
        
        # 合并静态特征
        if static_features is not None:
            features = np.concatenate([features, static_features], axis=1)
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 预测
        return self.model.predict(features_scaled)
    
    def save_model(self, filepath):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'wavelet': self.wavelet,
            'level': self.level
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        predictor = cls(wavelet=data['wavelet'], level=data['level'], use_grid_search=False)
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        return predictor


class PytorchWaveletSVM(nn.Module):
    """PyTorch包装的小波SVM模型，用于统一接口"""
    def __init__(
        self,
        input_dim=2,
        static_dim=0,
        wavelet='db4',
        level=3,
        kernel='rbf'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.static_dim = static_dim
        
        # 创建SVM模型
        self.svm_model = WaveletSVMPredictor(
            wavelet=wavelet,
            level=level,
            kernel=kernel,
            use_grid_search=False  # 训练时可以设置为True
        )
        
        self.is_fitted = False
    
    def fit(self, X, y, static_features=None):
        """训练模型"""
        # 转换为numpy数组
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if static_features is not None and isinstance(static_features, torch.Tensor):
            static_features = static_features.cpu().numpy()
        
        # 确保y是1D数组
        if len(y.shape) > 1:
            y = y.ravel()
        
        self.svm_model.fit(X, y, static_features)
        self.is_fitted = True
    
    def forward(self, x, static=None):
        """前向传播"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forward()")
        
        # 转换为numpy数组
        x_np = x.cpu().numpy()
        static_np = static.cpu().numpy() if static is not None else None
        
        # 预测
        predictions = self.svm_model.predict(x_np, static_np)
        
        # 转换回张量
        return torch.FloatTensor(predictions).to(x.device)