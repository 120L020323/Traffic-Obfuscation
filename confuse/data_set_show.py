import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def check_block_boundary(data1, data2):
    time_diff1 = float(data1)
    time_diff2 = float(data2)

    if time_diff1 > 1400 and (time_diff2 - time_diff1) > 0:
        return True
    else:
        return False


def generate_grayscale_images(input_folder, output_folder):
    file_list = sorted(os.listdir(input_folder))[:6]  # 获取前6个文件，并按文件名排序
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        x = [entry['length'] for entry in data]
        y = [entry['time_difference'] for entry in data]

        plt.subplot(2, 3, i + 1)
        plt.scatter(x, y, c='black', s=10, alpha=0.5)
        plt.title(f'Browsing {i + 1}')
        plt.xlabel('Length')
        plt.ylabel('Time Difference')

    plt.suptitle('Grayscale Images of Browsing Data')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Browsing_original.png'))
    plt.show()


# 生成直方图像
def generate_grayscale_images_thirty(input_folder, output_folder):
    file_list = sorted(os.listdir(input_folder))[:7]  # 获取前7个文件，并按文件名排序
    block_count = 0  # 记录块的数量
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        block_data = []
        current_block = []
        for i in range(len(data) - 1):
            current_block.append(data[i])
            if check_block_boundary(data[i]['time_difference'], data[i + 1]['time_difference']):
                if len(current_block) >= 100:  # 只保存元素数量大于等于 100 的块
                    block_data.append(current_block)
                current_block = []

        current_block.append(data[-1])  # 处理最后一个数据
        if len(current_block) >= 1000:  # 只保存元素数量大于等于 100 的块
            block_data.append(current_block)

        block_count += len(block_data)  # 更新块的数量

        for j, block in enumerate(block_data):
            x = [float(entry['length']) for entry in block]
            y = [float(entry['time_difference']) for entry in block]

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, c='black', s=10, alpha=0.5)
            # plt.title(f'Browsing {i + 1} Block {j + 1}')
            # plt.xlabel('Length')
            # plt.ylabel('Time Difference')
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}.png'))
            plt.close()

    print(f'Total number of blocks: {block_count}')


# 图像扩充，镜像，90°，270°
def generate_images_with_transformations(input_folder, output_folder):
    file_list = sorted(os.listdir(input_folder))[:7]  # 获取前7个文件，并按文件名排序
    block_count = 0  # 记录块的数量

    for i, file_name in enumerate(file_list):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        block_data = []
        current_block = []
        for i in range(len(data) - 1):
            current_block.append(data[i])
            if check_block_boundary(data[i]['time_difference'], data[i + 1]['time_difference']):
                if len(current_block) >= 100:  # 只保存元素数量大于等于 100 的块
                    block_data.append(current_block)
                current_block = []

        current_block.append(data[-1])  # 处理最后一个数据
        if len(current_block) >= 1000:  # 只保存元素数量大于等于 100 的块
            block_data.append(current_block)

        block_count += len(block_data)  # 更新块的数量

        for j, block in enumerate(block_data):
            x = [float(entry['length']) for entry in block]
            y = [float(entry['time_difference']) for entry in block]

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}.png'))

            # 镜像 x=750 轴
            x_mirror = [750 - entry for entry in x]
            plt.figure(figsize=(6, 4))
            plt.scatter(x_mirror, y, c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}_mirror.png'))

            # 旋转90°
            plt.figure(figsize=(6, 4))
            plt.scatter(y, x, c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}_90.png'))

            # 旋转270°
            plt.figure(figsize=(6, 4))
            plt.scatter(y, [-val for val in x], c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}_270.png'))

            # 镜像后旋转90°
            plt.figure(figsize=(6, 4))
            plt.scatter(y, x_mirror, c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}_mirror_90.png'))

            # 镜像后旋转270°
            plt.figure(figsize=(6, 4))
            plt.scatter(y, [-val for val in x_mirror], c='black', s=10, alpha=0.5)
            plt.savefig(os.path.join(output_folder, f'Browsing_Block_{j + 1}_mirror_270.png'))

            plt.close('all')


def plot_images_in_one_row(image_paths):
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    for i in range(2):
        for j in range(3):
            img = mpimg.imread(image_paths[i * 3 + j])
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.show()


def main():
    # 图片路径列表
    image_paths = [
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23.png",
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23_90.png",
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23_270.png",
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23_mirror.png",
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23_mirror_90.png",
        r"E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img\Browsing_Block_23_mirror_270.png"
    ]
    # 调用函数
    plot_images_in_one_row(image_paths)
    input_json_folder = r'E:\\tyy_confuse\\Pcaps\\Browsing_test\\Browsing_json'
    output_json_folder = r'E:\\tyy_confuse\\Pcaps\\Browsing_test\\Browsing_img'

    # generate_grayscale_images_thirty(input_json_folder, output_json_folder)
    folder_path = r'E:\tyy_confuse\Pcaps\Browsing_test\Browsing_img'
    # process_image_folder(folder_path)
    # generate_images_with_transformations(input_json_folder, output_json_folder)


if __name__ == '__main__':
    main()
