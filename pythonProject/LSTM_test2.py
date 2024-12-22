# 完整版LSTM，4种标签分类
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 初始化空列表用于存储训练和测试数据
data_train = []
data_test = []

# 定义包含数据文件的文件夹路径
input_dirs = [
    "E:/tyy_confuse/Pcaps/Browsing_test/Browsing_json",
    "E:/tyy_confuse/Pcaps/CHAT_test/CHAT_json",
    "E:/tyy_confuse/Pcaps/P2P_test/P2P_json",
    "E:/tyy_confuse/Pcaps/VOIP_test/VOIP_json"
]
# 定义标签与数字的对应关系
labels = {
    "Browsing": 0,
    "CHAT": 1,
    "P2P": 2,
    "VOIP": 3
}
# 遍历每个文件夹，读取并处理数据
for idx, input_dir in enumerate(input_dirs):
    folder_name = os.path.basename(input_dir).split('_')[0]  # 获取文件夹名，例如'Browsing'
    # 遍历文件夹中的所有文件
    for file in os.listdir(input_dir):
        # 打开并读取json文件中的数据
        with open(os.path.join(input_dir, file), 'r') as f:
            chat_data = json.load(f)
            # 为每个样本添加标签
            for sample in chat_data:
                sample['label'] = labels[folder_name]
            # 将处理后的数据添加到训练集中
            data_train.extend(chat_data)

# 从训练集中随机选择100000个样本
data_train = np.array(data_train)
np.random.shuffle(data_train)
data_train = data_train[:100000]

# 从训练集中提取特征和标签，并进行分割
X = np.array([[sample['length'], sample['time_difference']] for sample in data_train], dtype=np.float32)
y = np.array([sample['label'] for sample in data_train])
# 将训练集分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=100)

# 将数据和标签转换为PyTorch张量
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)


# 创建 LSTM 模型
class LSTMModel(nn.Module):
    # 初始化函数，定义网络层
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # 前向传播函数
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取序列最后一个时间步的输出
        return out


# 设置LSTM模型的参数
input_size = X_train.shape[1]
hidden_size = 32
num_layers = 5
output_size = len(labels)
# 实例化LSTM模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 设置训练参数并开始训练
num_epochs = 10
batch_size = 128
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    # 使用tqdm显示训练进度
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(num_batches):
            # 获取一个批次的数据和标签
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            # 训练步骤：前向传播、计算损失、反向传播、更新参数
            optimizer.zero_grad()
            outputs = model(X_train[start_idx:end_idx].unsqueeze(1))
            loss = criterion(outputs.squeeze(), y_train[start_idx:end_idx].long())

            # 添加 L2 正则化
            l2_lambda = 0.001  # 正则化系数
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

# 在验证集上评估模型性能
model.eval()

with torch.no_grad():  # 不需要计算梯度，节省内存和计算资源
    outputs = model(X_val.unsqueeze(1))
    _, predicted = torch.max(outputs.data, 1)
    accuracy1 = (predicted == y_val).sum().item() / len(y_val)
# 打印验证集上的准确率
print(f'Training Accuracy: {accuracy1}')

# 在单独的测试文件上进行测试，并计算各类别的概率
test_file = "E:/tyy_confuse/Pcaps/LSTM_test/VOIP_mimicry_training_add.json"
with open(test_file, 'r') as f:
    test_data = json.load(f)

# 处理测试文件数据
X_test = np.array([[sample['length'], sample['time_difference']] for sample in test_data], dtype=np.float32)
y_test = np.array([labels[sample['label']] for sample in test_data])  # 使用labels字典进行标签转换
# 只保留前10000个数据点
X_test = X_test[:100000]
y_test = y_test[:100000]

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 测试单独的测试文件
model.eval()
with torch.no_grad():
    outputs = model(X_test.unsqueeze(1))
    probabilities = nn.functional.softmax(outputs, dim=1)
    browsing_prob = probabilities[:, 0].mean().item()
    chat_prob = probabilities[:, 1].mean().item()
    p2p_prob = probabilities[:, 2].mean().item()
    voip_prob = probabilities[:, 3].mean().item()

# 打印各类别的平均概率
print(f'BROWSING Probability: {browsing_prob}')
print(f'CHAT Probability: {chat_prob}')
print(f'P2P Probability: {p2p_prob}')
print(f'VOIP Probability: {voip_prob}')
