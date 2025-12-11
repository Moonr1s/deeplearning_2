import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 步骤 1: 数据准备
# ==========================================
def load_and_process_data():
    # 1. 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 2. 数据划分 (训练集:验证集:测试集 = 6:2:2)
    # 先划分出20%作为测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 再将剩余的80%划分为训练集(75%)和验证集(25%) -> 0.8 * 0.25 = 0.2 (总体的20%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # 3. 数据归一化 (使用训练集的统计量)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. 转换为PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, iris.target_names

# ==========================================
# 步骤 2: 模型构建 (FNN)
# ==========================================
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        # 隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数
        self.relu = nn.ReLU()
        # 输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==========================================
# 步骤 3: 训练模型
# ==========================================
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=1000, learning_rate=0.01):
    # 定义损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义优化器：随机梯度下降 (SGD)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录训练损失
        train_losses.append(loss.item())
        
        # --- 验证阶段 ---
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                _, predicted = torch.max(val_outputs.data, 1)
                accuracy = (predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(accuracy)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}')
                
    return train_losses, val_accuracies

# ==========================================
# 步骤 4: 可视化
# ==========================================
def plot_results(train_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    # 绘制准确率曲线 (注意：验证集是每10个epoch记录一次)
    plt.subplot(1, 2, 2)
    plt.plot(range(10, len(train_losses) + 1, 10), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    
    plt.show()

# ==========================================
# 步骤 5 & 6: 模型评估与预测
# ==========================================
def evaluate_and_predict(model, X_test, y_test, target_names):
    # 5. 模型评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).sum().item() / y_test.size(0)
        
    print(f"\n最终测试集准确率 (Test Accuracy): {test_accuracy:.4f}")
    
    # 6. 模型预测 (随机抽取一个样本)
    print("\n--- 单个样本预测演示 ---")
    sample_idx = 0  # 选取测试集第一个样本
    sample_data = X_test[sample_idx]
    true_label = y_test[sample_idx].item()
    
    with torch.no_grad():
        # 增加一个维度 (batch_size=1)
        pred_logit = model(sample_data.unsqueeze(0))
        _, pred_label = torch.max(pred_logit, 1)
        pred_label = pred_label.item()
        
    print(f"样本特征 (Normalized): {sample_data.numpy()}")
    print(f"真实标签: {true_label} ({target_names[true_label]})")
    print(f"预测标签: {pred_label} ({target_names[pred_label]})")
    print(f"预测结果: {'正确' if true_label == pred_label else '错误'}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_and_process_data()
    print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}, 测试集样本数: {len(X_test)}")
    
    # 2. 构建模型
    input_dim = 4      # Iris特征数量
    hidden_dim = 16    # 隐藏层大小
    output_dim = 3     # 类别数量
    model = FeedForwardNN(input_dim, hidden_dim, output_dim)
    print(f"\n模型结构:\n{model}")
    
    # 3. 训练模型
    print("\n开始训练...")
    losses, accuracies = train_model(model, X_train, y_train, X_val, y_val)
    
    # 4. 可视化结果
    plot_results(losses, accuracies)
    
    # 5 & 6. 评估与预测
    evaluate_and_predict(model, X_test, y_test, class_names)