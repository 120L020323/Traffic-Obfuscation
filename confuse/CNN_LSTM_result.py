#  数据展示
import os
import matplotlib.pyplot as plt
import numpy as np

# 填入您的数据，按照名义顺序排列
CNN_Simulated = [
    [0.27, 0.10, 0.33, 0.16],
    [0.24, 0.37, 0.07, 0.23],
    [0.13, 0.20, 0.25, 0.15],
    [0.18, 0.27, 0.21, 0.35]
]

CNN_GAN = [
    [0.14, 0.03, 0.83, 0.00],
    [0.00, 0.07, 0.00, 0.78],
    [0.79, 0.00, 0.03, 0.08],
    [0.05, 0.86, 0.11, 0.09]
]

LSTM_Simulated = [
    [0.00, 0.04, 0.65, 0.00],
    [0.08, 0.11, 0.09, 0.72],
    [0.00, 0.71, 0.03, 0.02],
    [0.75, 0.00, 0.15, 0.00]
]

LSTM_GAN = [
    [0.00, 0.00, 0.73, 0.00],
    [0.76, 0.00, 0.00, 0.00],
    [0.17, 0.15, 0.09, 0.71],
    [0.00, 0.72, 0.13, 0.04]
]

# 设置标题
titles = ["CNN_Simulated", "CNN_GAN", "LSTM_Simulated", "LSTM_GAN"]

# 设置横纵坐标标签
labels = ["Browsing", "CHAT", "P2P", "VOIP"]

# 绘制图表和保存子图
for idx, data in enumerate([CNN_Simulated, CNN_GAN, LSTM_Simulated, LSTM_GAN]):
    # 设置画布和子图
    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(data, cmap='Blues')

    # 添加数值和坐标标签
    for x in range(4):
        for y in range(4):
            if data[x][y] > 0.5:
                text_color = 'white'  # 超过50%的色块用白色打印
            else:
                text_color = 'black'

            # 放大字号并且只打印百分号前的数字
            text = ax.text(y, x, f'{data[x][y] * 100:.0f}%',
                           ha="center", va="center", color=text_color, fontsize=16)

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels)

    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(labels)

    # 设置标题
    ax.set_title(titles[idx])

    # 添加色带
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percentage', rotation=-90, va="bottom")

    # 保存子图
    save_path = f"E:\\tyy_confuse\\result\\{titles[idx]}.png"
    plt.savefig(save_path)
