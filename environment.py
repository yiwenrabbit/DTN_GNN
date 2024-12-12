import random
import os
import Task
import Vehicle as VE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import numpy as np
import EdgeServer as eds
import Comm

x_axis = 1000  # 网络长 单位米
y_axis = 1000  # 网络宽
grid_size = 10  # 网格划分数量（每行/列的格子数）
time_slot = 0.1  # 时隙长度为0.1秒
vechiles_for_task = 4 #判断区域中车辆数量是否超过该变量，如果是则生成任务
subtasks_dag = 5      #每个任务的子任务数量



class Env:
    def __init__(self, edge_num, vehicle_num, load_from_file=False, vehicle_file_name="vehicle_positions.json", edge_file_name = 'edge_positions.json'):
        self.edge_num = edge_num
        self.vehicle_num = vehicle_num
        self.vehicle_file_name = vehicle_file_name
        self.edge_file_name = edge_file_name
        self.grid_points, self.horizontal_lines, self.vertical_lines = self.generate_grid()
        self.dt_edge_dic = self.init_dt_edge_dic()

        if load_from_file:
            self.vehicles = self.load_vehicle_positions()
            self.edges = self.load_edge_positions()
        else:
            self.vehicles = self.init_vehicles()
            self.edges = self.init_edges()
        self.tasks = self.generate_task_from_region()
        self.all_subtasks = self.get_all_subtasks()

    def get_all_subtasks(self):
        sub_tasks = []
        for task in self.tasks:
            for sub_task in task.subtasks:
                sub_tasks.append(sub_task)
        return sub_tasks

    ##########初始化Edge Server 放置DT的信息#####################
    def init_dt_edge_dic(self):
        dt_edge_dic = dict()
        for i in range(self.edge_num):
            dt_edge_dic[str(i)] = ""
        #print(dt_edge_dic)
        return dt_edge_dic

    #对DT的放置位置进行修改
    def change_dt_edge(self, edge_id, vec_id):
        if vec_id in self.dt_edge_dic[edge_id]:
            print('DT already on this edge,'+'edge_id:'+edge_id+'vec_id:'+vec_id)
        else:
            for key, value in self.dt_edge_dic.items():
                if isinstance(value, list) or isinstance(value, set):
                    # 如果值是列表或集合，移除指定值
                    if vec_id in value:
                        value.remove(vec_id)
            self.dt_edge_dic[edge_id].append(vec_id)

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

        self.save_edge_positions(edges)
        return edges

    def save_edge_positions(self, edges):
        """
        保存边缘服务器位置到文件
        """
        edge_data = [
            {
                "edge_id": edge.edge_id,
                "location": edge.location
            }
            for edge in edges
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

            velocity = random.uniform(10, 50)  # 随机分配速度
            vehicles.append(VE.Vec(vec_id=i, location=start_location, velocity=velocity, direction=direction))

            # 调试打印车辆初始化信息
            #print(f"Initialized Vehicle {i}: Location={start_location}, Velocity={velocity}, Direction={direction}")
        self.save_vehicle_positions()  # 保存初始位置到文件

        return vehicles

    ##########对网络分区便于产生任务################################
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
    #def store_DT_to_edge(self, vehicle_id, edge_id):

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
            vehicle.move(self.grid_points, self.horizontal_lines, self.vertical_lines, time_slot, x_axis, y_axis,grid_size)
            # 调试打印车辆位置
            #print(f"Vehicle {vehicle.vec_id}: New Position {vehicle.location}")

    ##########DT最近放置方法###############################
    def assign_dts_to_nearest_edges(self):
        """
        将车辆对应的 DT 分配到最近的 Edge 上，并更新 self.dt_edge_dic
        """
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

        print("Updated DT to Edge mapping:")
        for edge_id, dts in self.dt_edge_dic.items():
            print(f"Edge {edge_id}: {dts}")

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
            if len(dic)!=0:
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
                if len(self.dt_edge_dic[str(edge.edge_id)])!=0:
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

    ###########现在需要进行决策###############
    def offloading_decisions(self):
        current_ready_tasks = []
        for task in self.tasks:
            current_ready_tasks.append(task.get_ready_subtasks())

        #决策需包含传输功率与目标
        return current_ready_tasks






if __name__ == "__main__":
    env = Env(edge_num=16, vehicle_num=20, load_from_file=True)
    env.assign_dts_to_nearest_edges()
    ready_tasks = env.offloading_decisions()


    #env.visualize_local_computing(80)
