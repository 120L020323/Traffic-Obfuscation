import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


# 读取文件并计算概率分布的函数
def read_and_calculate_distribution(filename):
    lengths = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 检查是否为空行
                length = int(line)
                lengths.append(length)

    unique_lengths, counts = np.unique(lengths, return_counts=True)
    probabilities = counts / len(lengths) * 100  # 转换为百分比
    return unique_lengths, probabilities


# 绘制概率分布图的函数
def plot_distribution(filename, title, ax):
    lengths, probabilities = read_and_calculate_distribution(filename)
    ax.bar(lengths, probabilities, width=1, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Length')
    ax.set_ylabel('Probability Distribution (%)')
    ax.grid(True)


# 保存概率分布图为图片文件
def save_probability_distribution_plot(filename):
    browsing_path = 'E:\\tyy_confuse\\Pcaps\\Browsing_test\\Browsing_txt\\BROWSING_tor_test.txt'
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    plot_distribution(filename, 'test_confuse_probability', axs)
    plt.tight_layout()
    plt.savefig('test_confuse_probability.png')


# 主函数
def main():
    browsing_path = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\browsing_random.txt'
    chat_path = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\chat_random.txt'
    p2p_path = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\\P2P_random.txt'
    voip_path = 'E:\\tyy_confuse\\Pcaps\\random_confuse\\VOIP_random.txt'

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    plot_distribution(browsing_path, 'Browsing_random', axs[0, 0])
    # plot_distribution(chat_path, 'CHAT_random', axs[0, 1])
    # plot_distribution(p2p_path, 'P2P_random', axs[1, 0])
    # plot_distribution(voip_path, 'VOIP_random', axs[1, 1])

    plt.tight_layout()
    plt.show()

    # 保存概率分布图为图片文件
    save_probability_distribution_plot(browsing_path)


if __name__ == "__main__":
    main()
