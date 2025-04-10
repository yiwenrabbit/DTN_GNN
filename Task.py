import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

max_data_size=60
min_data_size=30
com_density = 1e7
min_delay = 15          #任务最小时延
max_delay = 20          #任务最大时延

class SubTask:
    """
    子任务类，代表一个任务中的具体处理单元
    """
    def __init__(self, task_id, subtask_index, vehicle_id, cpu_cycles=None, data_size=None):
        self.task_id = task_id
        self.subtask_index = subtask_index
        self.vehicle_id = vehicle_id
        self.subtask_id = f"{task_id}-{subtask_index}"  # 唯一的子任务 ID
        self.data_size = data_size if data_size else np.random.randint(min_data_size, max_data_size)  # 单位为 Mb
        # self.cpu_cycles = cpu_cycles if cpu_cycles else self.data_size * com_density  # 假设每 Mb 数据需要 1e7 个 CPU 周期
        # self.trans_size = self.data_size  # 如果需要传输，传输数据不为 0
        self.compute_size = self.data_size  # 如果需要处理数据，初始值为数据大小
        self.offloading_target = [0,1]      # 第一位：0代表是在edge server上算，1代表在车上算  第二位：0
        self.subtask_ready = 0              # 标识当前任务是否可以处理，可以为1否则为0
        self.subtask_done = 1               # 标识当前任务是否处理完成，完成为0否则为1
        self.off_tag = 1                    # 标识当前任务是否处于卸载状态,1表示没有，0表示正在卸载
        self.reward_given = False

    def compute_size_accu(self, accuracy):
        change = 0.5 * accuracy + 0.5
        self.compute_size = self.data_size * change


    def transmit_data(self, amount):
        """
        模拟传输指定数量的数据
        """
        if self.trans_size > 0:
            self.trans_size -= amount
            self.trans_size = max(self.trans_size, 0)  # 确保传输大小不为负值

    def compute_data(self, amount):
        """
        模拟处理指定数量的数据
        """
        #print(f"Subtask {self.subtask_index} of DT {self.vehicle_id}, Percentage:{1-self.compute_size/self.data_size}")
        self.compute_size -= amount
        self.compute_size = max(self.compute_size, 0)  # 确保计算大小不为负值
        if self.compute_size == 0:
            self.subtask_done = 0


    def is_completed(self):
        """
        判断子任务是否完成
        """
        return self.subtask_done == 0

    def __repr__(self):
        return (f"SubTask(task_id={self.task_id}, subtask_id={self.subtask_id}, vehicle_id={self.vehicle_id}, "
                f"cpu_cycles={self.cpu_cycles}, remaining_compute_size={self.compute_size})")


class Task:
    """
    任务类，包含多个子任务和任务依赖图
    """
    def __init__(self, task_id, n_subtasks, vehicles):
        self.task_id = task_id
        self.n_subtasks = n_subtasks
        self.vehicles = vehicles
        self.task_delay = np.random.uniform(min_delay, max_delay)
        self.original_delay = self.task_delay
        self.origin_task_delay = self.task_delay

        # 初始化子任务，确保 subtask_id 唯一
        self.subtasks = self.generate_unique_subtasks()

        self.dependencies = self.generate_custom_dag_dependencies()  # 生成任务依赖关系
        self.graph = self.build_dependency_graph()  # 构建任务依赖图
        self.original_graph = self.graph.copy()  # 保留完整的图
        self.is_completed = False  # 标志任务是否完成
        #自动计算传播中心性
        #self.centrality_scores = self.calculate_betweenness_centrality()
        # 初始化子任务的状态
        # 计算传播性
        #self.task_spread = self.calculate_task_spread()  # 新增：计算并存储传播性
        self.update_subtask_status()

    def calculate_task_spread(self):
        """
        计算每个子任务的传播性，结合：
        1. 出度 (out-degree)：任务影响的直接后续任务数
        2. PageRank：衡量全局传播性
        3. 关键路径长度：任务影响整个 DAG 任务完成的程度
        """
        if self.graph is None or len(self.graph.nodes) == 0:
            return {}

        # 计算每个子任务的出度中心性
        out_degree_centrality = nx.out_degree_centrality(self.graph)

        # 计算 PageRank 作为全局传播性指标
        pagerank_centrality = nx.pagerank(self.graph, alpha=0.85)

        # 计算关键路径长度
        longest_path_length = {}
        for node in self.graph.nodes:
            # 计算从当前节点到终点的最长路径
            longest_path_length[node] = nx.dag_longest_path_length(self.graph.subgraph(nx.descendants(self.graph, node).union({node})))

        # 归一化关键路径长度
        max_length = max(longest_path_length.values()) if longest_path_length else 1
        normalized_path_length = {node: val / max_length for node, val in longest_path_length.items()}

        # 计算最终传播性度量
        task_spread = {}
        for node in self.graph.nodes:
            # 动态权重分配
            weight_out_degree = 0.4  # 出度中心性权重
            weight_pagerank = 0.3   # PageRank 权重
            weight_path_length = 0.3  # 关键路径长度权重

            # 计算传播性
            task_spread[node] = (
                    weight_out_degree * out_degree_centrality[node] +
                    weight_pagerank * pagerank_centrality[node] +
                    weight_path_length * normalized_path_length[node]
            )

        return task_spread


    #计算中心度
    # def calculate_betweenness_centrality(self):
    #     """
    #     计算 DAG（self.graph）中 **每个子任务** 的传播中心性。
    #     返回一个字典 {subtask_id: centrality_value}
    #     """
    #     # 如果图为空或未正确生成，则返回空字典
    #     if self.graph is None or len(self.graph.nodes) == 0:
    #         return {}
    #
    #     # 计算每个子任务的传播中心性
    #     # betweenness_centrality() 返回的字典 {subtask_id: center_value}
    #     centrality = nx.betweenness_centrality(self.graph, normalized=True)
    #
    #     # 返回包含传播中心性的字典
    #     return centrality  # {subtask_id: 中心性数值}

    def generate_unique_subtasks(self):
        """
        确保每个车辆 ID 只选择一次生成子任务
        """
        used_vehicles = set()
        subtasks = []

        for i in range(self.n_subtasks):
            # 过滤掉已使用的车辆
            available_vehicles = [v for v in self.vehicles if v not in used_vehicles]
            if not available_vehicles:
                raise ValueError("Not enough unique vehicles to assign to all subtasks.")

            # 随机选择一个未使用的车辆
            vehicle_id = np.random.choice(available_vehicles)
            used_vehicles.add(vehicle_id)

            # 创建子任务
            subtask = SubTask(
                task_id=self.task_id,
                subtask_index=i,
                vehicle_id=vehicle_id
            )
            subtasks.append(subtask)

        return subtasks

    def generate_custom_dag_dependencies(self):
        """
        自定义生成任务间的依赖关系，确保生成的图是 DAG。
        """
        dependencies = []

        # 左侧的串行部分 (例如：0 -> 1 -> 2)
        for i in range(self.n_subtasks // 2 - 1):
            dependencies.append((self.subtasks[i].subtask_id, self.subtasks[i + 1].subtask_id))

        # 中间解锁后可并行的任务 (例如：2 -> 3 和 2 -> 4)
        parallel_start = self.subtasks[self.n_subtasks // 2 - 1].subtask_id
        parallel_nodes = [self.subtasks[i].subtask_id for i in range(self.n_subtasks // 2, self.n_subtasks - 1)]

        for node in parallel_nodes:
            dependencies.append((parallel_start, node))

        # 最终任务 (例如：3 -> 6, 4 -> 6)
        final_task = self.subtasks[-1].subtask_id
        for node in parallel_nodes:
            if np.random.rand() > 0.5:  # 随机决定是否连接到最终任务
                dependencies.append((node, final_task))

        # 确保最终任务至少有一个父节点
        if not any(dep[1] == final_task for dep in dependencies):
            dependencies.append((parallel_nodes[-1], final_task))

        return dependencies

    def build_dependency_graph(self):
        """
        构建任务依赖的图，使用 subtask_id 作为节点
        """
        G = nx.DiGraph()

        # 添加子任务的 subtask_id 作为节点
        G.add_nodes_from([subtask.subtask_id for subtask in self.subtasks])

        # 添加依赖关系
        G.add_edges_from(self.dependencies)

        # 检查是否是有向无环图
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Generated graph is not a DAG!")
        return G

    def get_ready_subtasks(self):
        """
        获取当前任务图中可以处理的子任务（即入度为 0 的节点）
        并更新子任务的状态。
        """
        ready_subtasks = []
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:  # 入度为 0
                subtask = next(st for st in self.subtasks if st.subtask_id == node)
                subtask.subtask_ready = 1  # 标记子任务为可以处理
                ready_subtasks.append(subtask)
        return ready_subtasks

    def update_subtask_status(self):
        """
        更新所有子任务的状态，设置 subtask_ready 和 subtask_done。
        """
        for subtask in self.subtasks:
            if subtask.is_completed():
                subtask.subtask_done = 0
            else:
                subtask.subtask_done = 1

        self.get_ready_subtasks()

    def update_task_status(self):
        """
        更新任务状态，移除已完成的子任务及其依赖关系。
        如果所有子任务完成，则标记任务完成。
        """
        # 收集已完成的子任务
        completed_tasks = [
            subtask.subtask_id for subtask in self.subtasks if subtask.is_completed()
        ]

        # 从图中移除已完成的子任务及其所有出边
        for subtask_id in completed_tasks:
            if self.graph.has_node(subtask_id):
                self.graph.remove_node(subtask_id)
                print(f"SubTask {subtask_id} completed and removed from the graph.")

        # 如果图中没有节点，标记任务完成
        if len(self.graph.nodes) == 0:
            self.is_completed = True
            print(f"Task {self.task_id} is completed!")
            return True
        else:
            return False

    def plot_dependency_graph(self, color="lightblue"):
        """
        绘制任务依赖图，节点标签使用每个子任务的 subtask_id
        """
        pos = nx.spring_layout(self.graph)  # 使用 spring 布局

        # 节点标签为每个子任务的 subtask_id
        labels = {node: node for node in self.graph.nodes}

        # 绘制图形
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_color=color, node_size=3000, font_size=10, font_weight="bold", font_color="black")

        plt.title(f"Dependency Graph for Task {self.task_id}")
        plt.show()

    #######获取所有任务的后续关联任务数据大小
    def get_descendant_subtasks(self, subtask):
        """
        获取指定子任务在 DAG 中的所有子节点任务
        """
        if not self.graph.has_node(subtask.subtask_id):
            raise ValueError(f"SubTask {subtask.subtask_id} is not in the graph.")

        # 使用 NetworkX 的 descendants 方法获取所有后续节点
        descendant_ids = nx.descendants(self.graph, subtask.subtask_id)

        # 找到所有对应的 SubTask 对象
        descendant_subtasks_size = [
            st.data_size for st in self.subtasks if st.subtask_id in descendant_ids
        ]
        #这里最后加的是任务当前剩余的任务量
        return sum(descendant_subtasks_size)+subtask.compute_size

    def __repr__(self):
        return f"Task(task_id={self.task_id}, n_subtasks={self.n_subtasks}, is_completed={self.is_completed})"


# vehicles = [1, 2, 3, 4, 5, 6]
# task = Task(task_id=1, n_subtasks=6, vehicles=vehicles)
#
# print("Initial Task Graph:")
# task.plot_dependency_graph()
#
# ready_subtasks = task.get_ready_subtasks()
# print("Ready Subtasks:", ready_subtasks)
#
# task.subtasks[0].compute_size = 0  # 模拟第一个子任务完成
# task.update_task_status()
#
# print("Updated Task Graph:")
# task.plot_dependency_graph()
#
# ready_subtasks = task.get_ready_subtasks()
# print("Ready Subtasks after Update:", ready_subtasks)