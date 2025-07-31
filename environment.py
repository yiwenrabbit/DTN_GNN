#env.py
import random
import os
import torch as T
import EdgeServer
import Task
import Vehicle as VE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import numpy as np
import EdgeServer as eds
import Comm
import GNN
import Digital_twin as DT
import pickle

x_axis = 1000  # 网络长 单位米
y_axis = 1000  # 网络宽
grid_size = 10  # 网格划分数量（每行/列的格子数）
time_slot = 0.1  # 时隙长度为0.1秒
vechiles_for_task = 4  # 判断区域中车辆数量是否超过该变量，如果是则生成任务
subtasks_dag = 5  # 每个任务的子任务数量
max_data_size = Task.max_data_size  # 子任务的大小定义在Task.py中
min_data_size = Task.min_data_size
com_density = Task.com_density  # 计算密度
max_edge_to_v_distance = 300  # 大于这个距离为1
min_edge_to_v_distance = 30  # 小于这个距离为0
dt_computing_power = 0.3e9  # DT在edge上会占据DT的算力
edge_max_cpu = EdgeServer.edge_max_cpu  # edge的初始算力
com_density = Task.com_density  # 计算密度

tran_power = Comm.power  # 27dbm 传输功率

min_delay = Task.min_delay  # 任务最小时延
max_delay = Task.max_delay  # 任务最大时延

min_accuracy = 0.5  # DT上传的数据百分比最小值
max_accuracy = 1

computing_probability = 0.8  # 车辆是算力车辆的可能性
edge_cpu_for_task = 0.5e9  # edge分配给每个任务的计算量

min_speed = 3
max_speed = 8  # 车辆速度10-30km/h

# reward 参数
time_over_due = -15
task_finish = 20
DT_update_reward = 50
w1 = 0.5  # 精度权重
w2 = 0.5  # 时延权重

step_reward_factor = 10

UPDATE_REWARD_SCALE = 0.2  # 建议从 0.05 开始测试，逐步调大

# =========mask 说明=======#
# ready_mask 1:任务现在可以处理， 0:任务还不能处理
# done_mask 1:还没有完成， 0:任务已经完成
# off_mask 1:表示任务没有在卸载，0: 表示任务在卸载


def ordering_subtasks_or_cpu(size, subtasks):  # 对于算力的排序第一个是处理数据量，第二个是id
    # 使用 zip 打包两个列表
    paired = list(zip(size, subtasks))

    # 按第一个列表的值排序，同时保持顺序稳定性
    sorted_paired = sorted(paired, key=lambda x: (-x[0], subtasks.index(x[1])))

    # 解压排序后的列表
    sorted_size, sorted_subtasks = zip(*sorted_paired)

    return sorted_size, sorted_subtasks


def calculate_process_data(vec_loc, edge_loc, vec_cpu):
    tran_rate = Comm.trans_rate(edge_loc, vec_loc, tran_power)
    time = (time_slot * vec_cpu) / (tran_rate * com_density + vec_cpu)
    processed_data_size = time * tran_rate
    return processed_data_size


# 计算真实距离
def calculate_real_distance(vec_loc, edge_loc):
    vec_x, vec_y = vec_loc
    edge_x, edge_y = edge_loc
    # 计算欧几里得距离
    distance = np.sqrt((vec_x - edge_x) ** 2 + (vec_y - edge_y) ** 2)

    if distance <= min_edge_to_v_distance:
        return min_edge_to_v_distance
    else:
        if distance <= max_edge_to_v_distance:
            return distance
        else:
            return -1


class Env:
    def __init__(self, edge_num, vehicle_num, load_from_file=False, vehicle_file_name="vehicle_positions.json",
                 edge_file_name='edge_positions.json'):
        self.edge_num = edge_num
        self.vehicle_num = vehicle_num
        self.vehicle_file_name = vehicle_file_name
        self.edge_file_name = edge_file_name
        self.grid_points, self.horizontal_lines, self.vertical_lines = self.generate_grid()
        self.dt_set = self.init_dt()
        self.subtask_accuracy_dict = {}
        self.last_update_step = {}  # {subtask_id: step}
        self.current_step = 0       # 当前的 step index
        self.subtask_accuracy_dict = {}        # 子任务精度
        self.last_update_step = {}             # 子任务最后更新时间步


        if load_from_file:
            self.vehicles = self.load_vehicle_positions()
            self.edges = self.load_edge_positions()
        else:
            self.vehicles = self.init_vehicles()
            self.save_vehicle_positions()  # 保存初始位置到文件
            self.edges = self.init_edges()
            self.save_edge_positions()

        self.tasks = self.generate_task_from_region()
        self.all_subtasks = self.get_all_subtasks()
        # DT放置初始化
        self.dt_edge_dic = self.init_dt_edge_dic()
        self.update_dt_in_step()
        self.edge_power_update()
        self.set_vehicle_cpu()  # 根据产生的任务设置车辆的算力
        self.task_spread_factor = self.tasks[0].calculate_task_spread()

    def get_all_subtasks(self):
        sub_tasks = []
        for task in self.tasks:
            for sub_task in task.subtasks:
                sub_tasks.append(sub_task)
        return sub_tasks

    ###########保存与加载环境###################################
    def save_env(self, file_name='./tmp/saved_env.pkl'):
        """
        保存当前 Env 对象到文件
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        print(f"Environment saved to {file_name}.")

    @staticmethod
    def load_env(file_name='./tmp/saved_env.pkl'):
        """
        从文件加载 Env 对象
        如果文件不存在，返回 None
        """
        try:
            with open(file_name, 'rb') as f:
                env = pickle.load(f)
            print(f"Environment loaded from {file_name}.")
            return env
        except FileNotFoundError:
            print(f"File {file_name} not found. Creating a new environment.")
            return None

    # 重置环境
    def reset_env(self):
        return self.load_env()

    ##########初始化DT#########################################
    def init_dt(self):
        dts = []
        for i in range(self.vehicle_num):
            new_dt = DT.DigitalT(i)
            dts.append(new_dt)
        return dts

    ##########初始化Edge Server 放置DT的信息
    def init_dt_edge_dic(self):
        dt_edge_dic = dict()
        for i in range(self.edge_num):
            dt_edge_dic[str(i)] = ""
        # print(dt_edge_dic)

        return dt_edge_dic

    # 对DT的放置位置进行修改
    def change_dt_edge(self, edge_id, vec_id):
        if vec_id in self.dt_edge_dic[edge_id]:
            print('DT already on this edge,' + 'edge_id:' + edge_id + 'vec_id:' + vec_id)
        else:
            for key, value in self.dt_edge_dic.items():
                if isinstance(value, list) or isinstance(value, set):
                    # 如果值是列表或集合，移除指定值
                    if vec_id in value:
                        value.remove(vec_id)
            self.dt_edge_dic[edge_id].append(vec_id)

    # DT最近放置方法
    def assign_dts_to_nearest_edges(self):
        """
        将车辆对应的 DT 分配到最近的 Edge 上，并更新 self.dt_edge_dic
        """
        self.dt_edge_dic = {key: [] for key in self.dt_edge_dic}

        for vehicle in self.vehicles:
            vehicle_location = np.array(vehicle.location)

            # 找到最近的 Edge
            nearest_edge = None
            min_distance = float('inf')
            for edge in self.edges:
                edge_location = np.array(edge.location)
                distance = np.linalg.norm(vehicle_location - edge_location)

                if distance < min_distance:
                    min_distance = distance
                    nearest_edge = edge

            # 将车辆的 DT 放到最近的 Edge 上
            if nearest_edge is not None:
                if self.dt_edge_dic[str(nearest_edge.edge_id)] == "":
                    self.dt_edge_dic[str(nearest_edge.edge_id)] = []
                self.dt_edge_dic[str(nearest_edge.edge_id)].append(vehicle.vec_id)

        # print("Updated DT to Edge mapping:")
        # for edge_id, dts in self.dt_edge_dic.items():
        #     print(f"Edge {edge_id}: {dts}")

    # 子任务的DT放置为空
    def assign_task_dts_to_null(self):
        for subtask in self.all_subtasks:
            vec_id = subtask.vehicle_id
            for key, value in self.dt_edge_dic.items():
                if isinstance(value, list) or isinstance(value, set):
                    # 如果值是列表或集合，移除指定值
                    if vec_id in value:
                        value.remove(vec_id)

    # 每一次决策都需要更新一次DT的放置
    def update_dt_in_step(self):
        self.assign_dts_to_nearest_edges()
        self.assign_task_dts_to_null()

    # 根据DT放置更新edge算力
    def edge_power_update(self):
        for key, value in self.dt_edge_dic.items():
            tmp_edge = self.edges[int(key)]
            tmp_edge.current_CPU = tmp_edge.CPU - len(value) * dt_computing_power

    # 根据dt的id查询对应放置的edge
    def find_dt_place(self, target_dt):
        dt_dic = self.dt_edge_dic
        # 查找包含目标值的键
        result_key = None
        for key, value_list in dt_dic.items():
            if target_dt in value_list:
                result_key = key
                break  # 找到后可以直接退出循环

        if result_key == None:
            print('dt is not placed')
        else:
            return result_key

    ##########初始化Edge Server################################
    def init_edges(self):
        edges = []
        step_x = x_axis / grid_size
        step_y = y_axis / grid_size

        points_per_row = int((grid_size + 1) ** 2 / self.edge_num)
        count = 0
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                if count >= self.edge_num:
                    break
                if (i * grid_size + j) % points_per_row == 0:
                    location = (i * step_x, j * step_y)
                    edge_server = eds.Edge(edge_id=count, location=location)
                    edges.append(edge_server)
                    count += 1

        return edges

    def save_edge_positions(self):
        """
        保存边缘服务器位置到文件
        """
        edge_data = [
            {
                "edge_id": edge.edge_id,
                "location": edge.location
            }
            for edge in self.edges
        ]
        with open(self.edge_file_name, "w") as f:
            json.dump(edge_data, f, indent=4)
        print(f"Saved initial edge server positions to {self.edge_file_name}")

    def load_edge_positions(self):
        """
        从文件加载边缘服务器位置
        """
        if not os.path.exists(self.edge_file_name):
            edges = self.init_edges()
        else:
            with open(self.edge_file_name, "r") as f:
                edge_data = json.load(f)

            edges = [
                eds.Edge(
                    edge_id=data["edge_id"],
                    location=tuple(data["location"])
                )
                for data in edge_data
            ]
            print(f"Loaded edge server positions from {self.edge_file_name}")
        return edges

    def file_exists(self, file_name):
        """
        检查边缘服务器文件是否存在
        """
        return os.path.exists(self.edge_file_name)

    # 返回当前edge的算力
    def get_edges_power(self):
        power = []
        edge_vec_power = np.zeros(self.edge_num)
        for index, edge in enumerate(self.edges):
            for vec in self.vehicles:
                vec_com = vec.local_computing
                if vec_com != 0:
                    distance = calculate_real_distance(vec.location, edge.location)
                    if distance <= max_edge_to_v_distance and distance != -1:
                        edge_vec_power[index] += vec_com / edge_max_cpu
            power.append(edge.current_CPU / edge_max_cpu)
        return np.array(power) + edge_vec_power

    def get_available_edges_power(self, distance):
        powers = self.get_edges_power()
        for i in range(len(powers)):
            if distance[i] == 0:
                powers[i] = 0
        return powers

    ##########初始化Vehicles################################
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
        with open(self.vehicle_file_name, "w") as f:
            json.dump(vehicle_data, f, indent=4)
        print(f"Saved initial vehicle positions to {self.vehicle_file_name}")

    def load_vehicle_positions(self):
        """
        从文件加载车辆位置
        """
        with open(self.vehicle_file_name, "r") as f:
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
        print(f"Loaded vehicle positions from {self.vehicle_file_name}")
        return vehicles

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

            velocity = random.uniform(min_speed, max_speed)  # 随机分配速度
            vehicles.append(VE.Vec(vec_id=i, location=start_location, velocity=velocity, direction=direction))

            # 调试打印车辆初始化信息
            # print(f"Initialized Vehicle {i}: Location={start_location}, Velocity={velocity}, Direction={direction}")

        return vehicles

    def get_vehicle_location(self, vehicle_id):
        """
        根据车辆 ID 获取车辆的位置
        参数:
            vehicle_id (int): 车辆的唯一 ID
        返回:
            tuple: (x, y) 车辆的位置坐标
        """
        for vehicle in self.vehicles:
            if vehicle.vec_id == vehicle_id:
                return vehicle.location
        raise ValueError(f"Vehicle ID {vehicle_id} not found")

    def set_vehicle_cpu(self):
        task_id = [subtask.vehicle_id for subtask in self.all_subtasks]
        for vehicle in self.vehicles:
            if vehicle.vec_id not in task_id:
                com_vehicle_probability = random.uniform(0, 1)
                if com_vehicle_probability <= computing_probability:
                    vehicle.set_vehicle_cpu()

    def calculate_distance(self):
        """
        计算每辆车与所有边缘服务器的距离。
        距离规则：
        - 超出范围的距离设置为 0。
        - 在范围内的距离归一化为 0 到 1，距离越近值越接近 1。
        """
        all_distances = []  # 用于存储所有车辆的距离信息

        for vehicle in self.vehicles:
            current_v_distance = []  # 当前车辆与所有边缘服务器的距离
            vehicle_x, vehicle_y = vehicle.location
            for edge in self.edges:
                # 解包坐标，确保计算的是标量
                edge_x, edge_y = edge.location

                # 计算欧几里得距离
                distance = np.sqrt((vehicle_x - edge_x) ** 2 + (vehicle_y - edge_y) ** 2)

                if distance > max_edge_to_v_distance:
                    # 距离超出范围，设为 0
                    relative_distance = 0
                else:
                    if distance < min_edge_to_v_distance:
                        # 距离小于最小值，设为 1
                        relative_distance = 1
                    else:
                        # 距离归一化为 0-1
                        relative_distance = 1 - (distance - min_edge_to_v_distance) / (
                                max_edge_to_v_distance - min_edge_to_v_distance)

                current_v_distance.append(relative_distance)  # 添加当前服务器的相对距离

            all_distances.append(current_v_distance)  # 添加当前车辆的所有距离

        return all_distances

    ##########对网络分区便于产生任务################################
    def divide_into_regions(self):
        """
        将网格划分为四个区域，并返回每个区域包含的车辆集合
        """
        # 定义区域集合
        regions = {
            "1": set(),  # 一个区域
        }

        # 遍历所有车辆，判断其位置属于哪个区域
        for vehicle in self.vehicles:
            x, y = vehicle.location
            regions["1"].add(vehicle.vec_id)
        return regions

    def generate_task_from_region(self):
        tasks = []
        regions = self.divide_into_regions()
        for region, vehicles in regions.items():
            if len(vehicles) >= vechiles_for_task:
                new_task = Task.Task(task_id=int(region), n_subtasks=subtasks_dag, vehicles=vehicles)
                tasks.append(new_task)
        return tasks

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

    # 更新剩余时间
    def task_delay_update(self):
        if self.tasks[0].task_delay <= time_slot:
            self.tasks[0].task_delay = 0
        else:
            self.tasks[0].task_delay -= time_slot

    ##########画图################################
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

    # def store_DT_to_edge(self, vehicle_id, edge_id):

    def display_network(self):
        """
        绘制整个网络，显示车辆和边缘服务器的位置，并用编号标注
        """
        plt.figure(figsize=(8, 8))
        plt.title("Network: Vehicles and Edge Servers")

        # 绘制网格
        for i in range(grid_size + 1):
            # 水平线
            plt.plot([0, x_axis], [i * y_axis / grid_size, i * y_axis / grid_size], 'gray', linewidth=0.5)
            # 垂直线
            plt.plot([i * x_axis / grid_size, i * x_axis / grid_size], [0, y_axis], 'gray', linewidth=0.5)

        # 绘制边缘服务器
        for edge in self.edges:
            x, y = edge.location
            plt.scatter(x, y, color='red', s=100,
                        label='Edge Server' if 'Edge Server' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(x + 10, y + 10, f"Edge {edge.edge_id}", fontsize=9, color='black', ha='center', va='center')

        # 绘制车辆
        for vehicle in self.vehicles:
            x, y = vehicle.location
            plt.scatter(x, y, color='blue', s=50,
                        label='Vehicle' if 'Vehicle' not in plt.gca().get_legend_handles_labels()[1] else "")
            # 在点旁标注编号，偏移一点位置避免重叠
            plt.text(x + 10, y + 10, str(vehicle.vec_id), fontsize=9, color='black', ha='center', va='center')

        # 图例
        plt.legend(loc='upper right')

        # 设置坐标范围和标签
        plt.xlim(0, x_axis)
        plt.ylim(0, y_axis)
        plt.xlabel("X Axis (m)")
        plt.ylabel("Y Axis (m)")
        plt.grid(True)
        plt.show()

    ##########车辆移动控制################################
    def move_vehicles(self):
        """
        让所有车辆移动一个时隙
        """
        for vehicle in self.vehicles:
            vehicle.move(self.grid_points, self.horizontal_lines, self.vertical_lines, time_slot, x_axis, y_axis,
                         grid_size)
            # 调试打印车辆位置
            # print(f"Vehicle {vehicle.vec_id}: New Position {vehicle.location}")

    ##########根据subtask_id取出该子任务#######################
    def get_subtask(self, subtask_id):
        for task in self.tasks:
            for sub_task in task.subtasks:
                if sub_task.subtask_id == subtask_id:
                    return sub_task
        # 如果没有找到，返回 None 或抛出异常
        return None

    ########## 本地计算 ##########
    def local_computing(self):

        current_subtasks = [subtask for task in self.tasks for subtask in task.get_ready_subtasks()]
        for edge in self.edges:
            dic = self.dt_edge_dic[str(edge.edge_id)]
            if len(dic) != 0:
                edge_tasks = [
                    subtask for subtask in current_subtasks
                    if subtask.vehicle_id in self.dt_edge_dic[str(edge.edge_id)]
                ]
                if edge_tasks:
                    computing_power = edge.CPU / len(edge_tasks)
                    computed_size = computing_power * time_slot / 1e7
                    for subtask in edge_tasks:
                        subtask.trans_size = 0
                        subtask.compute_data(computed_size)
        for task in self.tasks:
            task.update_task_status()
        self.move_vehicles()

    ########## 可视化本地计算 ##########
    def visualize_local_computing(self, time_slots):
        fig, ax = plt.subplots(figsize=(10, 10))

        def init():
            ax.clear()
            for i in range(grid_size + 1):
                ax.plot([0, x_axis], [i * y_axis / grid_size, i * y_axis / grid_size], 'gray', linewidth=0.5)
                ax.plot([i * x_axis / grid_size, i * x_axis / grid_size], [0, y_axis], 'gray', linewidth=0.5)
            ax.set_xlim(0, x_axis)
            ax.set_ylim(0, y_axis)

        def update(frame):
            ax.clear()
            init()

            # 每帧更新
            self.local_computing()

            # 绘制边缘服务器和子任务进度
            for edge in self.edges:
                x, y = edge.location
                ax.scatter(x, y, color='red', s=100)
                ax.text(x + 10, y + 10, f"Edge {edge.edge_id}", fontsize=9, color='black', ha='center', va='center')

                # 子任务完成进度
                if len(self.dt_edge_dic[str(edge.edge_id)]) != 0:
                    valid_subtasks = [subtask for subtask in self.all_subtasks
                                      if subtask.vehicle_id in self.dt_edge_dic[str(edge.edge_id)]]
                    progress_text = [
                        f"S{subtask.subtask_id}: {100 - (subtask.compute_size / subtask.data_size * 100 if subtask.data_size != 0 else 100):.1f}%"
                        for subtask in valid_subtasks
                    ]
                    if progress_text:
                        ax.text(x, y - 20, "\n".join(progress_text), fontsize=8, color='blue', ha='center', va='center')
            # 绘制车辆
            for vehicle in self.vehicles:
                x, y = vehicle.location
                ax.scatter(x, y, color='blue', s=50)
                ax.text(x + 10, y + 10, f"V{vehicle.vec_id}", fontsize=9, color='black', ha='center', va='center')

            ax.set_title(f"Local Computing Visualization - Time Slot {frame + 1}")

        ani = FuncAnimation(fig, update, init_func=init, frames=time_slots, interval=500, repeat=False)
        ani.save("local_computing_with_progress.gif", writer="pillow", dpi=80, savefig_kwargs={"facecolor": "white"})
        plt.show()

    def get_subtask(self, subtask_id):
        for task in self.tasks:
            for sub_task in task.subtasks:
                if sub_task.subtask_id == subtask_id:
                    return sub_task
        return None

    ############生成特征图####################
    def normalize_features(self, x):
        """
        对特征矩阵进行归一化
        参数:
            x (numpy.ndarray): 原始特征矩阵
        返回:
            numpy.ndarray: 归一化后的特征矩阵
        """
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()  # 使用 Min-Max 标准化
        return scaler.fit_transform(x)

    def get_gcn_inputs(self):
        """
        构建 GCN 的输入特征矩阵和边矩阵
        返回:
            x (torch.Tensor): 特征矩阵，形状为 [num_nodes, num_features]
            edge_index (torch.Tensor): 边矩阵，形状为 [2, num_edges]
        """
        import torch

        # 所有子任务的列表
        all_subtasks = self.get_all_subtasks()

        # 与所有edge的距离
        all_distance = self.calculate_distance()

        # 节点特征矩阵 x
        # 每个子任务的特征： [subtask_ready, normalized_data_size, flattened_distance, available_edge_power]
        raw_features = []
        for subtask in all_subtasks:
            dt = self.dt_set[subtask.vehicle_id]

            # 展平距离和功率特征，并组合成单个特征列表
            node_features = [
                subtask.subtask_ready,
                dt.current_update / dt.update_size,  # dt的更新情况
                subtask.data_size / max_data_size * subtask.subtask_done,  # 归一化的数据大小
            ]
            raw_features.append(node_features)

        # 将特征转换为 Tensor
        x = torch.tensor(raw_features, dtype=torch.float)

        # 边矩阵 edge_index
        edge_list = []
        for task in self.tasks:
            for dependency in task.dependencies:
                src, dst = dependency
                src_idx = next(i for i, st in enumerate(all_subtasks) if st.subtask_id == src)
                dst_idx = next(i for i, st in enumerate(all_subtasks) if st.subtask_id == dst)
                edge_list.append((src_idx, dst_idx))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # 转置为 [2, num_edges]

        return x, edge_index

    def generate_masks(self):
        ready_mask = []
        edge_mask = []
        task_id = []
        done_mask = []
        off_mask = []
        for subtask in self.all_subtasks:
            ready_mask.append(subtask.subtask_ready)
            task_id.append(int(subtask.vehicle_id))
            done_mask.append(subtask.subtask_done)
            off_mask.append(subtask.off_tag)

        distance = self.calculate_distance()

        for id in task_id:
            edge_mask.append(distance[id])
        edge_mask = [[1 if x != 0 else 0 for x in row] for row in edge_mask]

        return ready_mask, edge_mask, done_mask, off_mask

    ############生成全局观察####################
    def network_observation(self):
        net_obs = []

    ##############对action进行处理##############
    def process_actions(self, actions):
        sub_tasks = subtasks_dag
        num_edges = self.edge_num
        # 确保 actions 是 NumPy 数组
        tmp_actions = actions.cpu().detach().numpy() if isinstance(actions, T.Tensor) else np.array(actions)
        actions = tmp_actions[0]

        # 切割动作
        edge_logits = actions[:sub_tasks * num_edges]  # 前 sub_tasks * num_edges
        decision_probs = actions[sub_tasks * num_edges:]  # 最后 sub_tasks

        # 重组 edge_logits 为 [sub_tasks, num_edges]
        edge_logits = edge_logits.reshape(sub_tasks, num_edges)

        # 从每个任务的边缘节点分布中选择最大概率的索引
        edge_selection = np.argmax(edge_logits, axis=1)

        return edge_selection, decision_probs

    ##########分配计算资源####################

    def allocate_resources(self, offload_subtasks):
        offloaded_size = 0

        following_size = [self.tasks[0].get_descendant_subtasks(off_task) for off_task in
                          offload_subtasks]
        # subtasks 就是排序后的任务
        _, subtasks = ordering_subtasks_or_cpu(following_size, offload_subtasks)

        used_vec = []
        for subtask in subtasks:
            ava_cn_id = []
            process_data = []

            edge_id = int(self.find_dt_place(subtask.vehicle_id))

            edge_loc = self.edges[edge_id].location
            for vec in self.vehicles:
                if vec.local_computing == 0 or vec.vec_id in used_vec:
                    pass
                else:
                    distance = calculate_real_distance(vec.location, edge_loc)
                    if distance == -1:
                        pass
                    else:
                        ava_cn_id.append(vec.vec_id)
                        process_data.append(calculate_process_data(vec.location, edge_loc, vec.local_computing))

            # 计算在edge上能算多少
            edge_power = self.edges[edge_id].current_CPU
            if edge_power >= edge_cpu_for_task:
                edge_data_process = edge_cpu_for_task * time_slot / com_density
            else:
                edge_data_process = edge_power * time_slot / com_density
            process_data.append(edge_data_process)

            # -1表示在edge上完成
            ava_cn_id.append(-1)

            # 对可选的计算节点进行排序
            ordered_data_size, result_id = ordering_subtasks_or_cpu(process_data, ava_cn_id)

            # 选择结果
            choose = result_id[0]
            if choose == -1:
                # 如果选择的是在edge上算则更新edge的算力，需注意之后可能需要重置
                self.edges[edge_id].computing_power_cost(edge_cpu_for_task)
            else:
                used_vec.append(choose)

            # 更新任务数据量, 下面的函数会直接将完成任务的done 置为0表示完成
            subtask.compute_data(ordered_data_size[0])
            offloaded_size += ordered_data_size[0] / subtask.data_size

        return offloaded_size

    #########DT传输数据更新
    def dt_update_after_actions(self, edge_selection, decision_probs):

        update_size = 0
        #finish_reward = 0
        # 首先将所有DT位置以最近分配，并删除子任务的放置
        self.update_dt_in_step()

        for index, sub_task in enumerate(self.all_subtasks):
            # 根据动作进行放置
            self.dt_edge_dic[str(edge_selection[index])].append(sub_task.vehicle_id)

        # 根据放置的DT更新edge的算力
        self.edge_power_update()

        # 可以开始卸载的任务
        offload_subtasks = []

        # 更新DT的数据
        for index, sub_task in enumerate(self.all_subtasks):
            decision = decision_probs[index]
            print("decision", decision)
            # 首先判断任务是不是已经完成的任务
            if sub_task.subtask_done == 0:
                # 任务已经完成
                continue

            else:
                # 任务还没有完成
                if sub_task.subtask_ready == 0:
                    # 任务还不能处理
                    continue

                else:
                    # 任务现在可以处理
                    if sub_task.off_tag == 0:
                        # 任务正在卸载过程中
                        offload_subtasks.append(sub_task)
                    else:
                        Dt_id = sub_task.vehicle_id
                        dt = self.dt_set[Dt_id]
                        # 任务没有在卸载
                        if random.random() < decision:
                            # 表示任务结束dt更新
                            sub_task.off_tag = 0  # 卸载标识置0

                            dt.update_done = True  # DT更新完成标识

                            offload_subtasks.append(sub_task)

                            accuracy = dt.return_accuracy()
                            sub_task.compute_size_accu(accuracy)  # 根据DT的更新量决定卸载任务数据量

                            #finish_reward += (accuracy - 0.5) * step_reward_factor  # 根据精度给予奖励
                            self.last_update_step[sub_task.subtask_id] = self.current_step
                            self.subtask_accuracy_dict[sub_task.subtask_id] = accuracy
                        else:
                            # 表示任务继续更新dt
                            vec_dt_id = sub_task.vehicle_id
                            vec_loc = self.vehicles[vec_dt_id].location
                            edge_loc = self.edges[edge_selection[index]].location

                            # 在该时隙更新dt
                            # 1.计算车辆到edge的传输速率
                            # 2.计算更新的dt数据量
                            # 3.更改dt状态，如果已经达到要求
                            trans_rate = Comm.trans_rate(vec_loc, edge_loc, tran_power)
                            update_dt_size = trans_rate * time_slot
                            update_ratio = dt.dt_update(update_dt_size)
                            update_size += update_ratio

                            # 💾 更新记录

                            #update_size += ratio - time_slot * self.task_spread_factor[sub_task.subtask_id]
                            print(f"[DT Update Debug] subtask={sub_task.subtask_id}, ratio={ratio:.6f}, "
                                  f"update_size_contrib={ratio - time_slot * self.task_spread_factor[sub_task.subtask_id]:.6f}, "
                                  f"update_size_total={update_size:.6f}")


                            # 这个是达到100%DT更新的情况
                            if dt.update_done == True:
                                accuracy = dt.return_accuracy()
                                sub_task.compute_size_accu(accuracy)  # 根据DT的更新量决定卸载任务数据量
                                sub_task.off_tag = 0
                                #finish_reward += (accuracy - 0.8) * step_reward_factor  # 根据精度给予奖励
                                #reward_bonus = (accuracy - 0.5) * 4
                                #finish_reward += reward_bonus * step_reward_factor
                                self.last_update_step[sub_task.subtask_id] = self.current_step
                            else:
                                # ❗️即使还没完成，也记录下当前更新时间
                                self.last_update_step[sub_task.subtask_id] = self.current_step



        # 返回该时隙可以卸载的任务
        return offload_subtasks, update_size


        #############step函数######################
    def step(self, DT_actions, new_action=False):
        reward = 0
        edge_selection, completion_rates = self.process_actions(DT_actions)

        # 需要做卸载决策的子任务集合
        offload_subtasks, update_size = self.dt_update_after_actions(edge_selection, completion_rates)

        if offload_subtasks:
            offloaded_size = self.allocate_resources(offload_subtasks)
        else:
            offloaded_size = 0

        # reward += update_size + finish_reward

        #reward += UPDATE_REWARD_SCALE * update_size + finish_reward
        reward += UPDATE_REWARD_SCALE * update_size

        # 更新剩余时间
        self.task_delay_update()

        # 更新车辆位置
        self.move_vehicles()

        # 更新DAG图,并且将可以处理的任务的ready置1
        status = self.tasks[0].update_task_status()

        if status:
            print(f"Accuracy: {self.return_dt_accuracy()}")

        _ = self.tasks[0].get_ready_subtasks()

        # 时延约束到达，或者任务提前完成
        # if self.tasks[0].task_delay == 0 and self.tasks[0].is_completed == False:
        #     if reward > 0:
        #         reward += time_over_due
        if self.tasks[0].task_delay == 0 and self.tasks[0].is_completed == False:
            # 超时但未完成任务，给予惩罚
            remaining_ratio = self.tasks[0].get_remaining_ratio()

            penalty_weight = 1.2          # 惩罚放大系数
            penalty_exponent = 1.5        # 非线性放大
            overtime_penalty = -time_over_due * penalty_weight * (remaining_ratio ** penalty_exponent)

            reward += overtime_penalty


        else:
            if self.tasks[0].task_delay >= 0 and self.tasks[0].is_completed == True:
                # 最小精度和平均精度
                min_accuracy, ave_accuracy = self.return_dt_accuracy()
                delay = self.tasks[0].task_delay / self.tasks[0].original_delay
                final_reward = w1 * ave_accuracy + w2 * (1 - delay)
                #reward += final_reward * task_finish
                min_acc_penalty = (min_accuracy - 0.5) * 6

                reward += final_reward * task_finish + min_acc_penalty
                print(f"[Reward Debug] update_size={UPDATE_REWARD_SCALE * update_size:.2f}, "
                      f"final_reward={final_reward * task_finish:.2f}, "
                      f"min_acc_penalty={min_acc_penalty:.2f}, "
                      f"TOTAL={reward:.2f}")
        x, edges = self.get_gcn_inputs()
        network_states = self.get_network_state()
        ready, edge, done_mask, off_mask = self.generate_masks()


        print(f"[Reward Debug] step_reward={reward:.4f}")
        print(f"  update_size={update_size:.4f} (贡献={UPDATE_REWARD_SCALE * update_size:.4f})")
        #print(f"  finish_reward={finish_reward:.4f}")
        print(f"  overtime_penalty={locals().get('overtime_penalty', 0):.4f}")
        print(f"  final_reward={locals().get('final_reward', 0):.4f}")
        print(f"  min_acc_penalty={locals().get('min_acc_penalty', 0):.4f}")

        self.current_step += 1



        return reward, x, edges, network_states, ready, edge, done_mask, off_mask

    def apply_precision_gap_penalty(self, buffer, penalty_coeff=10.0):
        """
        在 episode 结束时，对低于平均精度的子任务所处时间步施加负向奖励惩罚
        参数:
            buffer: 存储每一步 reward 的经验池（例如 ReplayBuffer）
            penalty_coeff: 精度差惩罚的缩放因子
        """
        if len(self.subtask_accuracy_dict) == 0:
            return  # 无任务完成，不惩罚

        avg_precision = sum(self.subtask_accuracy_dict.values()) / len(self.subtask_accuracy_dict)

        for subtask_id, prec in self.subtask_accuracy_dict.items():
            if prec < avg_precision and subtask_id in self.last_update_step:
                gap = avg_precision - prec
                step_idx = self.last_update_step[subtask_id]

                # 如果经验池还未存储到这个步数，跳过
                if step_idx < len(buffer.reward_memory):
                    buffer.reward_memory[step_idx] -= penalty_coeff * gap
                    print(f"[Penalty Applied] subtask_id={subtask_id}, step={step_idx}, "
                          f"gap={gap:.4f}, penalty={penalty_coeff * gap:.4f}")



    def return_dt_accuracy(self):
        accuracy = []
        for subtask in self.all_subtasks:
            accuracy.append(self.dt_set[subtask.vehicle_id].return_accuracy())

        return min(accuracy), np.average(np.array(accuracy))

    def print_all_accuracy(self):
        res = []
        for subtask in self.all_subtasks:
            res.append(self.dt_set[subtask.vehicle_id].return_accuracy())
        return res

    # 获取网络状态####
    def get_network_state(self):
        # 任务时延归一化
        task_delay = self.tasks[0].task_delay / max_delay

        # 所有子任务的列表
        all_subtasks = self.get_all_subtasks()

        # 与所有 edge 的距离
        all_distance = self.calculate_distance()

        # 原始特征提取
        raw_features = []
        for subtask in all_subtasks:
            distance_features = all_distance[subtask.vehicle_id]  # 距离特征（已计算）
            available_power = self.get_available_edges_power(distance_features)  # 可用功率
            dt = self.dt_set[subtask.vehicle_id]

            # 组合单个子任务的特征
            node_features = [
                dt.current_update / dt.update_size,  # dt 的更新情况
                subtask.compute_size / max_data_size * subtask.subtask_done,  # 数据大小归一化
                *distance_features,  # 展平距离特征
                *available_power  # 展平功率特征
            ]
            raw_features.append(node_features)

        # 将 raw_features 转换为二维张量（[num_subtasks, feature_dim]）
        raw_features = T.tensor(raw_features, dtype=T.float)

        # 展平 raw_features（将 [num_subtasks, feature_dim] 转为 [num_subtasks * feature_dim]）
        flattened_features = raw_features.flatten()

        # 拼接当前任务时延
        network_state = T.cat([flattened_features, T.tensor([task_delay], dtype=T.float)])

        return network_state


if __name__ == "__main__":
    ####生成新的环境
    edge_num = 16
    vec_num = 20
    env = Env(edge_num, vec_num, True)
    env.save_env()

    # env = Env.load_env()
    # env.tasks[0].plot_dependency_graph()
    # ready, edge = env.generate_masks()
    # task = env.tasks[0]
    # offloading_tasks = [task.subtasks[2], task.subtasks[3], task.subtasks[1]]
    # size = [task.get_descendant_subtasks(of_task) for of_task in offloading_tasks]
    # ordered_sub = ordering_subtasks(size, offloading_tasks)
    # print([subtask.subtask_index for subtask in ordered_sub])

    ####测试计算每个时隙能处理的任务量
    # env = Env.load_env()
    # env.display_network()
    # vec_loc =  env.vehicles[5].location
    # edge_loc = env.edges[1].location
    # vec_cpu = env.vehicles[15].local_computing
    # print(calculate_process_data(vec_loc, edge_loc, vec_cpu))
    # print(edge_cpu_for_task * time_slot/com_density)

    # env.visualize_local_computing(80)
