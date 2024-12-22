import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

max_sequence_length = 10  # 设定一个最大序列长度


# 加载数据函数，进行填充或截断操作，并计算包长度的概率分布
def load_data(file_paths):
    data = []
    labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data_from_file = json.load(file)
            single_file_data = []

            # 计算每个数据集中数据包长度的概率分布
            packet_lengths = np.array([item['length'] for item in data_from_file])
            unique_lengths, length_counts = np.unique(packet_lengths, return_counts=True)
            total_packets = len(packet_lengths)
            length_probabilities = dict(zip(unique_lengths, length_counts / total_packets))

            # 忽略概率小于1%的数据
            length_probabilities_filtered = {k: v for k, v in length_probabilities.items() if v >= 0.01}

            for item in data_from_file:
                length = item['length']
                if length in length_probabilities_filtered:
                    probability = length_probabilities_filtered[length]
                    if len(single_file_data) < max_sequence_length:
                        single_file_data.append([length, probability, item['time_difference']])

            data.append(single_file_data)
            labels.append(data_from_file[0]['label'])

    # 对数据进行填充或截断操作，使其具有相同的长度
    padded_data = pad_sequences(data, maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')

    # 转换标签为独热编码
    labels = to_categorical(np.array(labels), num_classes=2)

    return padded_data, labels


# 加载训练集和测试集
train_files = ["chat_{}.json".format(i) for i in range(10)]
test_files = ["chat_10.json"]

# 初始化空列表
x_train = []
y_train = []

# 逐个加载训练集文件
for file in train_files:
    x, y = load_data([file])
    x_train.append(x)
    y_train.append(y)

# 将列表转换为 numpy 数组，并将形状调整为 (samples, timesteps, features)
x_train = np.array(x_train).reshape(len(train_files), -1, 3)
y_train = np.array(y_train)

# 加载并处理测试集数据
x_test, y_test = load_data(test_files)
x_test = np.array(x_test).reshape(len(test_files), -1, 3)
y_test = np.array(y_test)
print('x_test shape:',x_test)
print('y_test shape:',y_test)
print('x_train shape:',x_train)
print('y_train shape:',y_train)

# 创建CNN模型
model = Sequential()
model.add(Conv1D(filters=1, kernel_size=2, activation='relu', input_shape=(max_sequence_length, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=1, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))  # 将输出层单元数改为2，并使用softmax激活函数


# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 使用binary_crossentropy作为损失函数

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=10, validation_data=(x_test, y_test))

# 在训练集和测试集上计算准确率
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)




