import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel("d:/neural_network/BostonHousingData.xlsx")

# 检查数据基本信息
print("数据集形状：", data.shape)
print(data.head())

# 2. 可选：计算相关系数并可视化（可以根据相关性选择主要特征）
corr_matrix = data.corr()
print("各特征与 MEDV 的相关系数：")
print(corr_matrix['MEDV'].sort_values(ascending=False))

# 可以根据相关性筛选特征，例如：选择与 MEDV 相关系数绝对值大于 0.5 的特征（不包括 MEDV 自身）
selected_features = corr_matrix['MEDV'].drop('MEDV')
selected_features = selected_features[abs(selected_features) > 0.5].index.tolist()
print("选取的主要特征：", selected_features)

# 3. 数据预处理
# 分离特征和目标值
X = data[selected_features].values
y = data['MEDV'].values.reshape(-1, 1)

# 检查是否存在缺失值
print("特征缺失值数量：", np.sum(pd.isnull(X)))
print("目标缺失值数量：", np.sum(pd.isnull(y)))

# 数据归一化处理：标准化（均值为0，标准差为1）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. 划分训练集和测试集，前 450 条作为训练集，后 50 条作为测试集
X_train = X_scaled[:450]
y_train = y_scaled[:450]
X_test = X_scaled[-50:]
y_test = y_scaled[-50:]

# 转换为 PyTorch 的张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 5. 构建全连接神经网络模型
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]  # 依据选取的特征数
model = RegressionNN(input_dim)
print(model)

# 6. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. 训练模型
num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(8,5))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# 8. 模型测试与评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

# 如果需要，将预测结果反归一化回原始房价尺度
predictions_original = scaler_y.inverse_transform(predictions.numpy())
y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())

# 显示部分预测结果与真实值对比
df_result = pd.DataFrame({
    "Predicted": predictions_original.flatten(),
    "Actual": y_test_original.flatten()
})
print(df_result.head())
