import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SubTask:
    """
    子任务类，代表一个任务中的具体处理单元
    """
    def __init__(self, task_id, subtask_index, vehicle_id, cpu_cycles=None, data_size=None):
        self.task_id = task_id
        self.subtask_index = subtask_index
        self.vehicle_id = vehicle_id
        self.subtask_id = f"{task_id}-{subtask_index}"  # 唯一的子任务 ID
        self.data_size = data_size if data_size else np.random.randint(100, 500)  # 单位为 Mb
        self.cpu_cycles = cpu_cycles if cpu_cycles else self.data_size * 1e7  # 假设每 Mb 数据需要 1e7 个 CPU 周期
        self.trans_size = self.data_size  # 如果需要传输，传输数据不为 0
        self.compute_size = self.data_size  # 如果需要处理数据，初始值为数据大小
        self.offloading_target = [0,1]      # 第一位：0代表是在edge server上算，1代表在车上算  第二位：0

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
        if self.trans_size == 0:  # 只有在数据传输完成后才可以处理数据
            print(f"Task {self.task_id}, SubTask {self.subtask_id} is under computation!")
            self.compute_size -= amount
            self.compute_size = max(self.compute_size, 0)  # 确保计算大小不为负值

    def is_completed(self):
        """
        判断子任务是否完成
        """
        return self.compute_size == 0

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

        # 初始化子任务，确保 subtask_id 唯一
        self.subtasks = [
            SubTask(
                task_id=task_id,
                subtask_index=i,
                vehicle_id=np.random.choice(list(vehicles))
            )
            for i in range(n_subtasks)
        ]

        self.dependencies = self.generate_custom_dag_dependencies()  # 生成任务依赖关系
        self.graph = self.build_dependency_graph()  # 构建任务依赖图
        self.is_completed = False  # 标志任务是否完成

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
        """
        ready_subtasks = []
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:  # 入度为 0
                ready_subtasks.append(next(st for st in self.subtasks if st.subtask_id == node))
        return ready_subtasks

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

    def __repr__(self):
        return f"Task(task_id={self.task_id}, n_subtasks={self.n_subtasks}, is_completed={self.is_completed})"
