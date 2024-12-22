import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 读取数据
data_train = []
data_test = []

# 遍历文件夹并处理数据
input_dirs = [
    "E:/tyy_confuse/Pcaps/Browsing_test/Browsing_json",
    "E:/tyy_confuse/Pcaps/CHAT_test/CHAT_json",
    "E:/tyy_confuse/Pcaps/P2P_test/P2P_json",
    "E:/tyy_confuse/Pcaps/VOIP_test/VOIP_json"
]

labels = {
    "Browsing": 1,
    "CHAT": 2,
    "P2P": 3,
    "VOIP": 4
}

for idx, input_dir in enumerate(input_dirs):
    folder_name = os.path.basename(input_dir).split('_')[0]  # 获取文件夹名，例如'Browsing'
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), 'r') as f:
            chat_data = json.load(f)
            for sample in chat_data:
                sample['label'] = labels[folder_name]
            data_train.extend(chat_data)


# 取100000个样本作为训练集
data_train = np.array(data_train)
np.random.shuffle(data_train)
data_train = data_train[:100000]

# 分割训练集和测试集
X = np.array([[sample['length'], sample['time_difference']] for sample in data_train], dtype=np.float32)
y = np.array([sample['label'] for sample in data_train])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=100)

# 转换为 PyTorch 的 Tensor
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

# 创建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取序列最后一个时间步的输出
        #out = self.fc(out)
        #print(out.shape)
        return out



input_size = X_train.shape[1]
hidden_size = 32
num_layers = 5
output_size = len(labels) + 1  # 加一是因为标签从1开始

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 2
batch_size = 648
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    with tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i+1) * batch_size

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


# 测试验证集效果
model.eval()

with torch.no_grad():
    outputs = model(X_val.unsqueeze(1))
    _, predicted = torch.max(outputs.data, 1)
    accuracy1 = (predicted == y_val).sum().item() / len(y_val)

print(f'Training Accuracy: {accuracy1}')

# 测试单独的测试文件
test_file = "E:/tyy_confuse/Pcaps/LSTM_test/BROWSING_mimicry_training_add.json"
with open(test_file, 'r') as f:
    test_data = json.load(f)
# 处理测试文件数据
X_test = np.array([[sample['length'], sample['time_difference']] for sample in test_data], dtype=np.float32)
y_test = np.array([1 if sample['label'] == 'Browsing' else 2 for sample in test_data])
# 只保留前10000个数据点
X_test = X_test[:10000]
y_test = y_test[:10000]

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 测试单独的测试文件
model.eval()
with torch.no_grad():
    outputs = model(X_test.unsqueeze(1))
    _, predicted = torch.max(outputs.data, 1)
    accuracy2 = (predicted == y_test).sum().item() / len(y_test)

print(f'Test Set Accuracy: {accuracy2}')

