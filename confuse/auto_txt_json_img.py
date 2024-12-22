import subprocess
import os
import json
import numpy as np
import cv2

# 自动获取length和time difference
def auto_txt(pcap_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 构建 tshark 命令
    tshark_path = r'"D:\D:\LenovoQMDownload\Wireshark\tshark.exe"'
    command = f'{tshark_path} -r {{file}} -T fields -e frame.len -e frame.time_delta'

    # 遍历文件夹下的所有 PCAP 文件
    for file_name in os.listdir(pcap_folder):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(pcap_folder, file_name)

            # 生成对应的文件名
            output_file_name = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_training.txt')

            # 执行 tshark 命令并获取输出
            result = subprocess.run(command.format(file=file_path), shell=True, capture_output=True, text=False)

            # 将输出添加到列表中（注意使用字节解码）
            output_lines = result.stdout.decode().splitlines()

            # 将提取的数据保存到单独的 txt 文件中
            with open(output_file_name, 'w') as file:
                file.write('\n'.join(output_lines))

    print("Extraction and saving of data completed.")  # #
# time difference累加计算，30s计为1个块
def accumulate_time_difference(input_file, output_file):
    # 读取原始文件
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 初始化变量
    current_accumulated_time = 0
    previous_time_delta = 0

    # 处理每一行数据
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            time_delta = float(parts[1])

            # 判断是否超过 30s，超过则写入下一个分界
            if current_accumulated_time + time_delta > 30:
                current_accumulated_time = 0  # 下一个分界的第一个时间戳
                previous_time_delta = 0

            # 计算累计时间
            current_accumulated_time += time_delta

            # 将累计时间最大限制为 30s
            if current_accumulated_time > 30:
                current_accumulated_time = 30

            # 重写文件并累计时间
            with open(output_file, 'a') as new_file:
                new_file.write(f'{parts[0]}\t{current_accumulated_time}\n')

            previous_time_delta = current_accumulated_time
# 获取文件路径
def process_files_in_folder(input_folder, output_folder):
    # 遍历文件夹内所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_add.txt')

            # 对每个文件进行时间累计处理
            accumulate_time_difference(input_file, output_file)

    print("Time difference accumulation and rewriting completed.")
# 自动将累计后的txt映射后转为json格式数据集，标签为Browsing
def generate_json_files(input_json_folder, output_json_folder):
    # 遍历文件夹内所有txt文件
    for file_name in os.listdir(input_json_folder):
        if file_name.endswith('.txt'):
            input_file = os.path.join(input_json_folder, file_name)
            output_file = os.path.join(output_json_folder, f'{os.path.splitext(file_name)[0]}.json')

            data = []
            # 读取txt文件并生成json格式数据
            with open(input_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split('\t')
                    time_difference = float(parts[1])*50
                    length = parts[0]
                    entry = {
                        'length': length,
                        'time_difference': time_difference,
                        'label': 'VOIP'
                    }
                    data.append(entry)

            # 保存为json文件
            with open(output_file, 'w') as json_file:
                json.dump(data, json_file)

    print("JSON files generation completed.")
# 检查分块边界
def check_block_boundary(data1, data2):
    time_diff1 = float(data1)
    time_diff2 = float(data2)

    if time_diff1 > 1400 and (time_diff2 - time_diff1) > 0:
        return True
    else:
        return False
# 保存分块图像
def count_blocks_in_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    block_count = 0
    start_time_diff = data[0]['time_difference']
    for i in range(1, len(data)):
        if check_block_boundary(start_time_diff, data[i]['time_difference']):
            block_count += 1
            start_time_diff = data[i]['time_difference']

    return block_count



def main():
    pcap_folder = r'E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_pcap'
    input_folder = r'E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_training'
    output_folder = r'E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_training'
    input_json_folder = r'E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_add'
    output_json_folder = r'E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_json'

    #auto_txt(pcap_folder, output_folder)
    #process_files_in_folder(input_folder, output_folder)
    generate_json_files(input_json_folder, output_json_folder)

if __name__ == "__main__":
    main()
