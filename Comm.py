import numpy as np

G_t = 2      #发射增益
f = 2.4e9    #传输带宽中心频率
G_r = 2      #接收增益
noise_power_density = -174      #白噪声功率密度(dBm/Hz)
band_width = 1e7               #带宽 Hz
power = 30                      #0.5W功率27dbm

def calculate_received_power(trans_loc, receive_loc, trans_power):
    """
    计算接收功率
    :param trans_loc: 发射端位置 (x, y) 元组
    :param receive_loc: 接收端位置 (x, y) 元组
    :param trans_power: 发射功率 (dBm)
    :return: 接收功率 (dBm)
    """
    # 计算发射与接收之间的距离
    trans_loc = np.array(trans_loc)
    receive_loc = np.array(receive_loc)
    distance = np.linalg.norm(trans_loc - receive_loc)

    if distance == 0:
        raise ValueError("Transmitter and receiver cannot be at the same location.")

    # 计算路径损耗 L (dB)
    L = 20 * np.log10(distance) + 20 * np.log10(f) - 147.55

    # 计算接收功率 Pr (dBm)
    received_power = trans_power + G_t + G_r - L

    return received_power


#################不考虑干扰计算传输速率########################
def calculate_transmission_rate(received_power):
    """
    计算传输速率（bps）

    :param received_power: 接收功率 (dBm)
    :param bandwidth: 带宽 (Hz)
    :param noise_power_density: 噪声功率密度 (dBm/Hz)
    :return: 传输速率 (bps)
    """
    # 将 dBm 转换为线性值 (W)
    Pr_linear = 10 ** ((received_power - 30) / 10)  # dBm -> W
    N0_linear = 10 ** ((noise_power_density - 30) / 10)  # dBm/Hz -> W/Hz

    # 计算 SNR（线性值）
    SNR = Pr_linear / (N0_linear * band_width)

    # 计算传输速率
    transmission_rate = band_width * np.log2(1 + SNR)  # bps

    return transmission_rate
def trans_rate(ts_loc, re_loc, tran_power):
    receive_power = calculate_received_power(ts_loc, re_loc, tran_power)
    rate = calculate_transmission_rate(receive_power)/1e6
    return rate             #Mb/s

