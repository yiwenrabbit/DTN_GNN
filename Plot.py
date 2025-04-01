import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    从文件读取数据，转换为 x, y 列表
    """
    try:
        with open(file_path, 'r') as file:
            y = [float(line.strip()) for line in file]  # 读取每一行并转换为浮点数
        x = np.arange(len(y))  # x 轴索引
        return x, np.array(y)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def plot_data(x, y, label, color):
    """
    绘制原始数据
    """
    if x is None or y is None:
        return
    plt.plot(x, y, label=label, color=color)

# 文件路径和参数
file = "./data/reward_16_20_5.txt"
loss_file = "./data/loss_16_20_5.txt"
x, y = read_data(file)

# 绘制曲线
plot_data(x, y, label="loss", color="red")


# 添加标题和标签
plt.title("reward Plot")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()
