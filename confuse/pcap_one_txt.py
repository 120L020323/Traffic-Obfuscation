import subprocess
import numpy as np
import matplotlib.pyplot as plt
import chardet
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体
font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=14)
matplotlib.use('Agg')
tshark_path = r'C:\Program Files\Wireshark\tshark.exe'
pcap_file_path = r'E:/tyy_confuse/Pcaps/random_confuse/VOIP_mimicry.pcap'
output_file_path = r'E:/tyy_confuse/Pcaps/random_confuse/VOIP_mimicry.txt'

# 执行命令行命令
cmd = [tshark_path, '-r', pcap_file_path, '-T', 'fields', '-e', 'frame.len']
output = subprocess.check_output(cmd, universal_newlines=True)

# 将输出结果保存到txt文件
with open(output_file_path, 'w') as f:
    f.write(output)

print(f"成功保存输出结果到文件：{output_file_path}")
# 以二进制模式读取文件内容，通过命令行获取的包长度原始数据
with open('E:/tyy_confuse/Pcaps/random_confuse/VOIP_mimicry.txt', 'rb') as file:
    rawdata = file.read()

# 使用chardet来检测编码方式
encoding_result = chardet.detect(rawdata)
print(encoding_result)

# 使用检测到的编码方式来解码文件内容，并仅处理非空行
packet_lengths = []
for line in rawdata.decode(encoding_result['encoding']).splitlines():
    line = line.strip()
    if line:
        packet_lengths.append(int(line))

# 计算直方图
X = 10
bin_edges = np.arange(min(packet_lengths), max(packet_lengths) + 2, X)
hist, _ = np.histogram(packet_lengths, bins=len(bin_edges)-1)
hist = hist/len(packet_lengths)*100

# 绘制概率分布图
plt.bar(bin_edges[:-1], hist, width=X)
plt.xlabel('Packet Length')
plt.ylabel('Probability (%)')
plt.title('confused')
plt.savefig('单个概率分布测试图.png')  # 保存为图片文件