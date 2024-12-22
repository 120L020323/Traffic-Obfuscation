import scapy.all as scapy
import numpy as np

# 读取原始pcap文件
original_pcap_path = r"E:\\tyy_confuse\\Pcaps\\random_confuse\\CHAT_test.pcap"
packets = scapy.rdpcap(original_pcap_path)

# 分析原始包长度
original_lengths = [len(packet) for packet in packets]
mean_length = np.mean(original_lengths)
std_dev = np.std(original_lengths)

# 设置要生成的包数量
num_packets = len(packets)

# 生成符合正态分布的离散包长度列表
# 确保长度不会小于最小包长度
mimicry_lengths = np.clip(np.round(np.random.normal(mean_length, std_dev, num_packets)),
                          a_min=min(original_lengths),
                          a_max=max(original_lengths)).astype(int)


# 修改包长度
modified_packets = []
for packet, new_length in zip(packets, mimicry_lengths):
    # 创建一个新的数据包，从原始数据包中复制所有层
    new_packet = scapy.Raw(load=packet.load)

    # 遍历原始数据包的每一层，并将它们添加到新数据包中
    for layer in packet.layers():
        # 获取层的原始数据字节
        try:
            layer_bytes = layer.__class__.__bytes__(layer)
        except AttributeError:
            # 如果没有__bytes__()方法，则尝试使用str()方法
            try:
                layer_bytes = bytes(str(layer), 'utf-8')
            except Exception as e:
                print(f"Error converting layer to bytes: {e}")
                continue

        # 如果层的原始数据字节超过新长度，则截断它
        if len(layer_bytes) > new_length:
            layer_bytes = layer_bytes[:new_length]

        # 将截断后的原始数据字节重新添加到层中
        layer.remove_payload()
        layer.add_payload(layer_bytes)

        new_packet = new_packet / layer

    # 确保新数据包的总长度等于所需长度
    # 如果不够，则添加填充
    if len(new_packet) < new_length:
        padding = scapy.Raw(load=b'\x00' * (new_length - len(new_packet)))
        new_packet = new_packet / padding

    # 将修改后的数据包添加到列表中
    modified_packets.append(new_packet)

# 保存修改后的pcap文件
mimicry_pcap_path = r"E:\\tyy_confuse\\Pcaps\\random_confuse\\CHAT_mimicry.pcap"
scapy.wrpcap(mimicry_pcap_path, modified_packets)

print("混淆后的pcap文件已保存至:", mimicry_pcap_path)
