# 拟态混淆
import numpy as np
import chardet
from matplotlib.font_manager import FontProperties
import os
import struct
from scapy.all import rdpcap, wrpcap, Ether,IP,TCP,Raw

# 设置Matplotlib的后端为Agg,允许matplotlib生成图像而不显示图形界面。
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

# 以二进制模式读取文件内容，通过命令行获取的包长度原始数据
with open('E:\\tyy_confuse\\Pcaps\\random_confuse\\BROWSING_test.txt', 'rb') as file:
    rawdata = file.read()

# 使用chardet来检测编码方式
encoding_result = chardet.detect(rawdata)
print(encoding_result)

# 二进制数据解码为字符串，非空行解析添加到packet_lengths列表
packet_lengths = []
for line in rawdata.decode(encoding_result['encoding']).splitlines():
    line = line.strip()
    if line:
        packet_lengths.append(int(line))

# 计算packet_lengths直方图
X = 4
bin_edges = np.arange(min(packet_lengths), max(packet_lengths) + 2, X)
hist, _ = np.histogram(packet_lengths, bins=len(bin_edges)-1)
hist = hist/len(packet_lengths)*100

# 对称的 D' 分布
diff = 800 - np.mean(packet_lengths)
mirror_packet_lengths = [800 + diff - length for length in packet_lengths]
hist_mirror, _ = np.histogram(mirror_packet_lengths, bins=len(bin_edges)-1)
hist_mirror = hist_mirror/len(packet_lengths)*100

# 根据给定的P查询对应的L包长度范围，10%以下误差不超过1%，10%以上误差不超过5%
def find_length_probability(hist, bin_edges, target_probability):
    lengths = []
    probabilities = hist / np.sum(hist)

    if target_probability <= 0.1:
        target_prob_min = target_probability - 0.01
        target_prob_max = target_probability + 0.01
    else:
        target_prob_min = target_probability - 0.05
        target_prob_max = target_probability + 0.05

    for i in range(len(bin_edges)-1):
        prob = probabilities[i]
        if target_prob_min <= prob <= target_prob_max:
            lengths.append(bin_edges[i])

    return lengths

# 假设要查询概率为P%时对应的包长度
target_probability = 0.05
lengths_for_target_probability = find_length_probability(hist, bin_edges, target_probability)
lengths_for_target_probability_mirror = find_length_probability(hist_mirror, bin_edges, target_probability)

if lengths_for_target_probability:
    print(f"概率恰好为{target_probability*100}%对应的D包长度为：{lengths_for_target_probability}")
else:
    print(f"找不到概率恰好为{target_probability*100}%对应的D包长度")

if lengths_for_target_probability_mirror:
    print(f"概率恰好为{target_probability*100}%对应的D'包长度为：{lengths_for_target_probability_mirror}")
else:
    print(f"找不到概率恰好为{target_probability*100}%对应的D'包长度")

# 原始pcap文件路径
original_pcap_file = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\P2P_test.pcap'
# 加载原始pcap文件
packets = rdpcap(original_pcap_file)

# 对pcap文件进行混淆,构造新数据包添到列表
confused_packets = []
for packet in packets:
    if packet.haslayer(TCP) and not packet.haslayer(Raw):
        L = len(packet)
        # 根据L和L_mirror进行分段或堆叠操作
        for L_mirror in lengths_for_target_probability_mirror:
            if L > L_mirror:
                # 分段操作
                num_segments = L // L_mirror
                for i in range(num_segments):
                    new_packet = packet.copy()
                    new_packet.load = packet.load[i*L_mirror:(i+1)*L_mirror]
                    confused_packets.append(new_packet)
            elif L < L_mirror:
                # 堆叠操作
                num_stacks = L_mirror // L
                new_packet = packet.copy()
                #只在TCP层进行负载堆叠
                new_payload = b""
                for i in range(num_stacks):
                    new_payload += bytes(packet[TCP].payload)
                new_packet[TCP].remove_payload()
                new_packet[TCP].add_payload(new_payload)

                confused_packets.append(new_packet)
    else:
        # 非TCP数据包或包含负载的数据包，直接添加到列表中
        confused_packets.append(packet.copy())

# 设置混淆数据包的链路类型
for new_packet in confused_packets:
    if new_packet.haslayer(Ether):
        link_type = new_packet[Ether].type
        new_packet[Ether].type = link_type

# 保存新的pcap文件
confused_pcap_file = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\BROWSING_mimicry.pcap'
wrpcap(confused_pcap_file, confused_packets)

print(f"成功生成混淆后的pcap文件：{confused_pcap_file}")
