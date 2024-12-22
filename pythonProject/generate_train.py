import json

# 从txt文件中读取数据包长度分布
length_distribution = {}
with open('E:\\tyy_confuse\\confuse\\Audio_packet_lengths.txt', 'r', encoding='utf-16') as f:
    for line in f:
        length = int(line.strip())
        if length in length_distribution:
            length_distribution[length] += 1
        else:
            length_distribution[length] = 1

# 计算概率分布
total_packets = sum(length_distribution.values())
length_probabilities = {length: count / total_packets for length, count in length_distribution.items()}

# 将训练集数据保存为JSON文件
training_data = {"length_probabilities": length_probabilities}
with open("Audio_training_data.json", "w") as f:
    json.dump(training_data, f, indent=4)

print("训练集已保存到 Audio_training_data.json 文件中。")
