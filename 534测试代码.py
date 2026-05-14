import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re

# ===================== 1. 基础配置（与训练代码一致） =====================
# 设备自动适配（GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备：{device}")

# 复用训练时的MAC地址识别函数（确保AP列筛选逻辑一致）
def is_mac_address(col_name):
    """匹配MAC地址格式（支持 xx:xx:xx:xx:xx:xx 或 xx-xx-xx-xx-xx-xx）"""
    mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
    return bool(mac_pattern.match(str(col_name).strip()))

# ===================== 2. 加载测试数据（与训练代码列筛选逻辑一致） =====================
# 测试数据路径（确保与文件位置匹配）
test_file_path = "534(已处理).xlsx"  # 若文件不在代码目录，替换为绝对路径（如 r"C:\xxx\534(已处理).xlsx"）

# 读取Excel（依赖openpyxl，无则执行：pip install openpyxl）
df_test = pd.read_excel(test_file_path, engine="openpyxl")
print(f"测试数据读取成功：共{len(df_test)}行数据，{len(df_test.columns)}列特征")

# 筛选AP列（与训练时完全一致：优先MAC列，兜底排除label/z轴）
ap_cols_test = [col for col in df_test.columns if is_mac_address(col)]
if len(ap_cols_test) == 0:
    exclude_keywords = ["label", "标签", "z", "Z", "z轴", "Z轴", "备注", "说明"]
    ap_cols_test = [col for col in df_test.columns if not any(kw in str(col).lower() for kw in exclude_keywords)]
print(f"测试数据中识别到AP信号列：{len(ap_cols_test)}个")

# 筛选坐标列（仅X/Y，排除AP列+label/z轴，与训练一致）
exclude_cols_test = ap_cols_test + [col for col in df_test.columns if any(kw in str(col).lower() for kw in ["label", "z", "Z"])]
coord_cols_test = [col for col in df_test.columns if col not in exclude_cols_test]

# 校验坐标列（确保为X/Y两维，与训练时的2维输出匹配）
if len(coord_cols_test) != 2:
    raise ValueError(f"测试数据需仅含X/Y 2个坐标列，当前检测到{len(coord_cols_test)}个：{coord_cols_test}\n请检查测试数据格式")
print(f"测试数据坐标列：{coord_cols_test}（已排除label和z轴）")

# 数据清洗（过滤RSSI正值异常值，与训练逻辑一致）
df_test_clean = df_test[(df_test[ap_cols_test] <= 0).all(axis=1)].copy()
print(f"测试数据清洗结果：原始{len(df_test)}行 → 有效{len(df_test_clean)}行")
if len(df_test_clean) == 0:
    raise ValueError("清洗后无有效测试数据，请检查AP信号列是否均为负值（RSSI正常值为负）")

# ===================== 3. 加载训练好的标准化器和模型 =====================
# 加载标准化器（关键：仅用transform，不重新fit，确保与训练数据分布一致）
scaler_path = "534数据集_标准化器.pkl"  # 训练时保存的标准化器路径
try:
    scaler = joblib.load(scaler_path)
    print("标准化器加载成功")
except FileNotFoundError:
    raise FileNotFoundError(f"未找到标准化器文件：{scaler_path}\n请确保该文件与测试代码在同一目录，或修改路径")

# 加载模型（与训练时完全一致的MAC_AP_Locator类）
class MAC_AP_Locator(nn.Module):
    def __init__(self, ap_count):
        super(MAC_AP_Locator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        
        # 自动计算全连接层维度（与训练时逻辑一致）
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, ap_count)
            conv_out = self.pool(self.relu(self.conv1(dummy_input)))
            conv_out = self.pool(self.relu(self.conv2(conv_out)))
            self.flatten_dim = conv_out.numel()
        
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 初始化模型并加载权重
model_path = "534数据集_Locator模型.pth"  # 训练时保存的模型权重路径
try:
    # 模型输入维度=测试数据的AP列数量
    model = MAC_AP_Locator(ap_count=len(ap_cols_test)).to(device)
    # 加载权重（兼容单卡/多卡训练的权重格式）
    state_dict = torch.load(model_path, map_location=device)
    # 处理多卡训练时的"module."前缀（若有）
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    # 切换为评估模式（关闭Dropout，固定批量归一化参数）
    model.eval()
    print("模型权重加载成功，已切换至评估模式")
except FileNotFoundError:
    raise FileNotFoundError(f"未找到模型权重文件：{model_path}\n请确保该文件与测试代码在同一目录，或修改路径")

# ===================== 4. 测试数据预处理（与训练一致） =====================
# 分离特征（AP信号）和真实标签（X/Y坐标）
X_test = df_test_clean[ap_cols_test].values  # 测试特征
y_test_true = df_test_clean[coord_cols_test].values  # 真实坐标

# 特征标准化（仅transform，与训练数据分布对齐）
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量（适配1D-CNN输入格式：[批次, 通道数, 特征数]）
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)
y_test_true_tensor = torch.tensor(y_test_true, dtype=torch.float32).to(device)

# ===================== 5. 模型推理（测试预测） =====================
print("\n开始模型推理...")
# 关闭梯度计算（加速推理+节省显存）
with torch.no_grad():
    y_test_pred_tensor = model(X_test_tensor)
    # 转换为numpy数组（方便后续计算误差和保存）
    y_test_pred = y_test_pred_tensor.cpu().numpy()
    y_test_true_np = y_test_true_tensor.cpu().numpy()

print("推理完成，开始计算测试精度...")

# ===================== 6. 测试精度评估（定位任务核心指标） =====================
# 计算定位误差（欧氏距离：每个样本的预测坐标与真实坐标的距离，单位：米）
test_distances = np.sqrt(np.sum((y_test_pred - y_test_true_np) ** 2, axis=1))

# 核心精度指标
metrics = {
    "平均定位误差(MAE)": np.mean(test_distances),
    "定位误差中位数": np.median(test_distances),
    "最大定位误差": np.max(test_distances),
    "最小定位误差": np.min(test_distances),
    "定位误差标准差": np.std(test_distances),
    "X坐标平均绝对误差": mean_absolute_error(y_test_true_np[:, 0], y_test_pred[:, 0]),
    "Y坐标平均绝对误差": mean_absolute_error(y_test_true_np[:, 1], y_test_pred[:, 1]),
    "X坐标均方根误差(RMSE)": np.sqrt(mean_squared_error(y_test_true_np[:, 0], y_test_pred[:, 0])),
    "Y坐标均方根误差(RMSE)": np.sqrt(mean_squared_error(y_test_true_np[:, 1], y_test_pred[:, 1]))
}

# 打印测试精度结果
print("\n" + "="*60)
print("534(已处理).xlsx 测试精度报告")
print("="*60)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:20s}: {metric_value:.4f} 米")
print("="*60)

# ===================== 7. 测试结果保存（Excel文件，方便分析） =====================
# 构建结果DataFrame（包含原始数据+预测坐标+定位误差）
result_df = df_test_clean.copy()
# 添加预测坐标列
result_df[f"预测_{coord_cols_test[0]}"] = y_test_pred[:, 0]
result_df[f"预测_{coord_cols_test[1]}"] = y_test_pred[:, 1]
# 添加定位误差列
result_df["定位误差_米"] = test_distances
# 添加误差等级（可选，方便快速筛选）
result_df["误差等级"] = pd.cut(
    result_df["定位误差_米"],
    bins=[0, 1, 3, 5, np.inf],
    labels=["优秀(<1m)", "良好(1-3m)", "一般(3-5m)", "较差(>5m)"]
)

# 保存结果到Excel
save_result_path = "534数据集_测试结果.xlsx"
result_df.to_excel(save_result_path, index=False, engine="openpyxl")

print(f"\n测试结果已保存至：{save_result_path}")
print(f"结果文件包含：原始数据、预测X/Y坐标、定位误差、误差等级")

# 打印误差等级分布（快速了解整体表现）
error_distribution = result_df["误差等级"].value_counts().sort_index()
print("\n定位误差等级分布：")
for level, count in error_distribution.items():
    percentage = (count / len(result_df)) * 100
    print(f"  {level}: {count}个样本 ({percentage:.1f}%)")