import random

import Task
import Vehicle as VE  # 确保 Vehicle.py 定义了 Vec 类
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import pandas as pd

x_axis = 1000  # 网络长 单位米
y_axis = 1000  # 网络宽
grid_size = 10  # 网格划分数量（每行/列的格子数）
time_slot = 0.1  # 时隙长度为0.1秒
vechiles_for_task = 4 #判断区域中车辆数量是否超过该变量，如果是则生成任务
subtasks_dag = 5      #每个任务的子任务数量


class Env:
    def __init__(self, edge_num, vehicle_num, load_from_file=False, file_name="vehicle_positions.json"):
        self.edge_num = edge_num
        self.vehicle_num = vehicle_num
        self.file_name = file_name
        self.grid_points, self.horizontal_lines, self.vertical_lines = self.generate_grid()

        if load_from_file:
            self.vehicles = self.load_vehicle_positions()
        else:
            self.vehicles = self.init_vehicles()
            self.save_vehicle_positions()  # 保存初始位置到文件

        self.tasks = self.generate_task_from_region()

    def save_vehicle_positions(self):
        """
        将车辆的初始位置保存到文件
        """
        vehicle_data = [
            {
                "vec_id": vehicle.vec_id,
                "location": vehicle.location,
                "velocity": vehicle.velocity,
                "direction": vehicle.direction,
            }
            for vehicle in self.vehicles
        ]
        with open(self.file_name, "w") as f:
            json.dump(vehicle_data, f, indent=4)
        print(f"Saved initial vehicle positions to {self.file_name}")

    def load_vehicle_positions(self):
        """
        从文件加载车辆位置
        """
        with open(self.file_name, "r") as f:
            vehicle_data = json.load(f)

        vehicles = [
            VE.Vec(
                vec_id=data["vec_id"],
                location=tuple(data["location"]),
                velocity=data["velocity"],
                direction=data["direction"],
            )
            for data in vehicle_data
        ]
        print(f"Loaded vehicle positions from {self.file_name}")
        return vehicles

    def divide_into_regions(self):
        """
        将网格划分为四个区域，并返回每个区域包含的车辆集合
        """
        # 定义区域集合
        regions = {
            "1": set(),  # 左上
            "2": set(),  # 右上
            "3": set(),  # 左下
            "4": set()  # 右下
        }

        # 遍历所有车辆，判断其位置属于哪个区域
        for vehicle in self.vehicles:
            x, y = vehicle.location
            if x <= x_axis / 2 and y > y_axis / 2:
                regions["1"].add(vehicle.vec_id)
            elif x > x_axis / 2 and y > y_axis / 2:
                regions["2"].add(vehicle.vec_id)
            elif x <= x_axis / 2 and y <= y_axis / 2:
                regions["3"].add(vehicle.vec_id)
            elif x > x_axis / 2 and y <= y_axis / 2:
                regions["4"].add(vehicle.vec_id)

        return regions

    def generate_task_from_region(self):
        tasks = []
        regions = self.divide_into_regions()
        for region, vehicles in regions.items():
            if len(vehicles) >= vechiles_for_task:
                new_task = Task.Task(task_id=int(region), n_subtasks=subtasks_dag, vehicles= vehicles)
                tasks.append(new_task)
        return tasks
    def display_vehicles(self):
        """
        打印所有车辆的位置、速度和方向，格式化输出
        """
        vehicle_data = [
            {
                "Vehicle ID": vehicle.vec_id,
                "Location": vehicle.location,
                "Velocity": vehicle.velocity,
                "Direction": vehicle.direction,
            }
            for vehicle in self.vehicles
        ]
        df = pd.DataFrame(vehicle_data)
        print(df)

    def plot_vehicle_positions(self):
        """
        绘制所有车辆的位置，并用编号标注
        """
        plt.figure(figsize=(8, 8))
        plt.title("Vehicle Positions")

        # 绘制网格
        for i in range(grid_size + 1):
            # 水平线
            plt.plot([0, x_axis], [i * y_axis / grid_size, i * y_axis / grid_size], 'gray', linewidth=0.5)
            # 垂直线
            plt.plot([i * x_axis / grid_size, i * x_axis / grid_size], [0, y_axis], 'gray', linewidth=0.5)

        # 绘制车辆
        for vehicle in self.vehicles:
            x, y = vehicle.location
            plt.scatter(x, y, color='blue', s=50)  # 蓝色点表示车辆
            # 在点旁标注编号，偏移一点位置避免重叠
            plt.text(x + 10, y + 10, str(vehicle.vec_id), fontsize=9, color='black', ha='center', va='center')

        plt.xlim(0, x_axis)
        plt.ylim(0, y_axis)
        plt.xlabel("X Axis (m)")
        plt.ylabel("Y Axis (m)")
        plt.grid(True)
        plt.show()

    def generate_grid(self):
        """
        生成网格点的坐标，以及水平线和垂直线上的所有坐标
        """
        grid_points = []
        horizontal_lines = []
        vertical_lines = []

        step_x = x_axis / grid_size
        step_y = y_axis / grid_size

        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                grid_points.append((i * step_x, j * step_y))  # 添加交点

                # 添加水平线上的点
                if i < grid_size:
                    horizontal_lines.append((i * step_x, j * step_y))

                # 添加垂直线上的点
                if j < grid_size:
                    vertical_lines.append((i * step_x, j * step_y))

        return grid_points, horizontal_lines, vertical_lines

    def init_vehicles(self):
        """
        初始化车辆位置，车辆随机选择在线上的一个点，并分配速度和方向
        """
        vehicles = []
        for i in range(self.vehicle_num):
            if random.random() < 0.5:  # 随机选择一条水平线
                start_location = random.choice(self.horizontal_lines)
                direction = random.choice(["left", "right"])
            else:  # 随机选择一条垂直线
                start_location = random.choice(self.vertical_lines)
                direction = random.choice(["up", "down"])

            velocity = random.uniform(10, 50)  # 随机分配速度
            vehicles.append(VE.Vec(vec_id=i, location=start_location, velocity=velocity, direction=direction))

            # 调试打印车辆初始化信息
            print(f"Initialized Vehicle {i}: Location={start_location}, Velocity={velocity}, Direction={direction}")

        return vehicles

    def move_vehicles(self):
        """
        让所有车辆移动一个时隙
        """
        for vehicle in self.vehicles:
            vehicle.move(self.grid_points, self.horizontal_lines, self.vertical_lines, time_slot, x_axis, y_axis,grid_size)
            # 调试打印车辆位置
            print(f"Vehicle {vehicle.vec_id}: New Position {vehicle.location}")

    def animate(self, frames=100, interval=100):
        """
        使用动画展示车辆在网格中的移动
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        def init():
            # 初始化画布内容
            self.ax.clear()
            for i in range(grid_size + 1):
                self.ax.plot([0, x_axis], [i * y_axis / grid_size, i * y_axis / grid_size], 'gray', linewidth=0.5)
                self.ax.plot([i * x_axis / grid_size, i * x_axis / grid_size], [0, y_axis], 'gray', linewidth=0.5)
            self.ax.set_xlim(0, x_axis)
            self.ax.set_ylim(0, y_axis)

        def update(frame):
            print(f"Updating frame {frame}...")
            self.move_vehicles()
            self.ax.clear()
            for i in range(grid_size + 1):
                self.ax.plot([0, x_axis], [i * y_axis / grid_size, i * y_axis / grid_size], 'gray', linewidth=0.5)
                self.ax.plot([i * x_axis / grid_size, i * x_axis / grid_size], [0, y_axis], 'gray', linewidth=0.5)

            for vehicle in self.vehicles:
                x, y = vehicle.location
                self.ax.scatter(x, y, color='blue', s=50)
                self.ax.text(x, y, f"{vehicle.vec_id}", fontsize=8, ha='center', va='center')

            self.ax.set_xlim(0, x_axis)
            self.ax.set_ylim(0, y_axis)
            self.ax.set_title("Vehicle Movement in Grid")

        ani = FuncAnimation(self.fig, update, init_func=init, frames=frames, interval=interval)

        # 保存动画为 GIF
        ani.save("vehicle_animation.gif", writer="pillow", dpi=80, savefig_kwargs={"facecolor": "white"})
        plt.show()


if __name__ == "__main__":
    edge_num = 16  # 假设有16个边缘节点
    vehicle_num = 20  # 假设有20辆车

    # 加载已有车辆位置
    load_from_file = True  # 设置为 False 以重新生成车辆并保存

    #env = Env(edge_num=edge_num, vehicle_num=vehicle_num)
    env = Env(edge_num=edge_num, vehicle_num=vehicle_num, load_from_file=load_from_file)
    env.display_vehicles()
    env.plot_vehicle_positions()

    for task in env.tasks:
        task.plot_dependency_graph()


    # 动画展示车辆在网格中的移动
    #env.animate(frames=100, interval=100)
