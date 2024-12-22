# 随机混淆
import random
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
from scapy.all import get_working_ifaces
from scapy.all import sniff


def get_link_type():
    # 使用 sniff 函数捕获一个数据包
    capture = sniff(count=1)
    # 获取捕获的数据包的链路类型
    link_type = capture[0].default_fields.get("linktype")
    if link_type is not None:
        return link_type
    else:
        print("Error: 'linktype' field not found in the captured packet.")
        return None


# 确保给定数据包列表中的数据包具有相同链路类型
def unify_link_type(packets):
    common_link_type = get_link_type()
    for packet in packets:
        if packet.default_fields.get("linktype") != common_link_type:
            packet.default_fields["linktype"] = common_link_type


# 随机增加或减少数据包的有效载荷长度
def randomize_packet_length(packet):
    # 定义最大增加和减少的长度
    MAX_INCREASE = 1500
    MAX_DECREASE = 100
    # 获取数据包的有效载荷层
    if Raw in packet:
        load_layer = packet.getlayer(Raw)
    else:
        load_layer = None
        # 如果有效载荷层存在且包含数据
    if load_layer and load_layer.load:
        # 随机选择增加或减少长度
        if random.choice([True, False]):
            increase_length = random.randint(200, max(200, min(MAX_INCREASE, len(load_layer.load))))
            load_layer.load += b'\0' * increase_length
        else:
            decrease_length = random.randint(10, min(MAX_DECREASE, len(load_layer.load)))
            if decrease_length < len(load_layer.load):
                load_layer.load = load_layer.load[:-decrease_length]
            else:
                load_layer.load = b''
        # 更新数据包长度字段
        packet_len = len(load_layer.load) + len(load_layer)
        if IP in packet:
            packet = packet[IP]
            packet.len = packet_len
        if TCP in packet:
            packet = packet[TCP]
            packet.len = packet_len - len(packet[TCP].payload)
        elif UDP in packet:
            packet = packet[UDP]
            packet.len = packet_len - len(packet[UDP].payload)
    return packet


# 读取给定pcap文件处理数据包,处理后写入新的pcap文件。
def process_pcap(input_pcap_path, output_pcap_path):
    packets = rdpcap(input_pcap_path)

    unify_link_type(packets)

    modified_packets = [randomize_packet_length(packet) for packet in packets]

    wrpcap(output_pcap_path, modified_packets)


# 在调用 process_pcap 之前先调用 get_working_ifaces() 来获取链路类型信息
get_working_ifaces()
input_pcap_path = "E:\\tyy_confuse\\Pcaps\\VOIP_test\\VOIP_pcap\\VOIP_test.pcap"
output_pcap_path = "E:\\tyy_confuse\\Pcaps\\random_confuse\\VOIP_test_random.pcap"
process_pcap(input_pcap_path, output_pcap_path)
