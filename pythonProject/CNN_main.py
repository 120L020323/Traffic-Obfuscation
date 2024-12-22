# CNN 分类：1.Audio；2.Browsing；3.chat；4.file-transfer；5.mail；6.P2P-Tor；7.VIDEO；8.待定
# -*- coding:utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 生成随机训练数据
X_train = np.random.rand(1000, 10, 1)  # 1000个样本，每个样本有10个时间步，每个时间步有1个特征
y_train = np.random.randint(2, size=(1000, 1))  # 1000个样本的二分类标签
# 生成随机测试数据
X_test = np.random.rand(200, 10, 1)  # 200个样本，每个样本有10个时间步，每个时间步有1个特征
y_test = np.random.randint(2, size=(200, 1))  # 200个样本的二分类标签
import json


# 将训练数据和测试数据转换为JSON格式
def save_data_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data.tolist(), f)

    # 保存训练数据


save_data_to_json(X_train, 'X_train.json')
save_data_to_json(y_train, 'y_train.json')

# 保存测试数据
save_data_to_json(X_test, 'X_test.json')
save_data_to_json(y_test, 'y_test.json')
# 创建模型
model = Sequential()

# 添加一维卷积层，32个卷积核，每个卷积核大小为2
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(10, 1)))

# 添加最大池化层，池化窗口大小为2
model.add(MaxPooling1D(pool_size=2))

# 将多维输入展平为一维，以便连接到全连接层
model.add(Flatten())

# 添加全连接层，128个节点，使用ReLU激活函数
model.add(Dense(128, activation='relu'))

# 添加输出层，使用sigmoid激活函数进行二分类
model.add(Dense(1, activation='sigmoid'))

# 编译模型，使用二元交叉熵作为损失函数，优化器为Adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 输出模型结构
model.summary()

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 在训练集和测试集上计算准确率
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)
