import subprocess
import os

# 定义 PCAP 文件夹路径
pcap_folder = r'E:\\tyy_confuse\\Pcaps\\chat_test\\chat_pcap'

# 构建 tshark 命令
tshark_path = r'"C:\Program Files\Wireshark\tshark.exe"'
command = f'{tshark_path} -r {{file}} -T fields -e frame.len -e frame.time_delta'

# 遍历文件夹下的所有 PCAP 文件
for index, file_name in enumerate(os.listdir(pcap_folder)):
    if file_name.endswith('.pcap'):
        file_path = os.path.join(pcap_folder, file_name)

        # 执行 tshark 命令并获取输出
        result = subprocess.run(command.format(file=file_path), shell=True, capture_output=True, text=False)

        # 将输出添加到列表中（注意使用字节解码）
        output_lines = result.stdout.decode().splitlines()

        # 生成对应的文件名
        output_file_name = f'chat_{index}_training.txt'

        # 将提取的数据保存到单独的 txt 文件中
        with open(output_file_name, 'w') as file:
            file.write('\n'.join(output_lines))
