# 获取txt文件和表示概率分布
import numpy as np
import matplotlib.pyplot as plt
import chardet
from matplotlib.font_manager import FontProperties

# 设置Matplotlib的后端为Agg
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

# 以二进制模式读取文件内容，通过命令行获取的包长度原始数据
with open('E:\\tyy_confuse\\confuse\\packet_lengths_confuse.txt', 'rb') as file:
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
X=4
bin_edges = np.arange(min(packet_lengths), max(packet_lengths) + 2, X)  # 以X为步长，柱子边界，也就是包括的长度分组
bin_test = len(bin_edges)-1
hist, _ = np.histogram(packet_lengths, bins=len(bin_edges)-1)
hist = hist/len(packet_lengths)*100

# 计算平均值和标准差
mean = np.mean(packet_lengths)
std = np.std(packet_lengths)

# 对称的 D' 分布
diff = 800 - mean
mirror_packet_lengths = [800 + diff - length for length in packet_lengths]
hist_mirror, _ = np.histogram(mirror_packet_lengths, bins=len(bin_edges)-1)
hist_mirror = hist_mirror/len(packet_lengths)*100

# 创建两个图形窗口和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# 在第一个子图中绘制原始概率分布
ax1.bar(bin_edges[:-1], hist, width=X, align='edge', alpha=0.5,label='Original probability distribution',color='blue')
ax1.set_xlabel('包长度', fontproperties=font)
ax1.set_ylabel('概率%', fontproperties=font)
ax1.set_title('混淆后D：Audio_tor长度概率分布', fontproperties=font)
ax1.legend()

# 在第二个子图中绘制对称的概率分布
ax2.bar(bin_edges[:-1], hist_mirror, width=X, align='edge', alpha=0.5, label='Modified probability distribution',color='red')
ax2.set_xlabel('包长度', fontproperties=font)
ax2.set_ylabel('概率%', fontproperties=font)
ax2.set_title('混淆后D‘：对称的Audio_tor长度概率分布', fontproperties=font)
ax2.legend()

# 调整子图之间的间距
plt.tight_layout()

# 保存图像到文件
plt.savefig('Aduio_length_distribution.png')



