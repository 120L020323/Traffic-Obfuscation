import json

# 定义分类阈值
threshold = 0.00001

# 读取数据并生成标签
try:
    with open('chat_10_training.txt.', 'r') as file:  # 假设您的txt文件名为data.txt
        data = []
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            length, time_difference = float(parts[0]), float(parts[1])
            # 根据时间差生成二分类标签
            label = 1 if time_difference > threshold else 0
            data.append({'length': length, 'time_difference': time_difference, 'label': label})

            # 将数据保存为JSON文件
    with open('chat_10.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("二分类标签已保存为 chat_10.json 文件。")

except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")



