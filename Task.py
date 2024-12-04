import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 子任务类
class SubTask:
    def __init__(self, task_id, subtask_id, cpu_cycles=None, data_size=None):
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.cpu_cycles = cpu_cycles if cpu_cycles else np.random.randint(500, 1500)
        self.data_size = data_size if data_size else np.random.randint(100, 500)

    def process_data(self, amount):
        """
        处理指定数量的数据
        :param amount: 要处理的数据量
        """
        self.data_size -= amount
        if self.data_size < 0:
            self.data_size = 0  # 避免数据量为负值

    def is_completed(self):
        """
        判断子任务是否完成
        :return: 如果剩余数据量为0，返回 True
        """
        return self.data_size == 0

    def __repr__(self):
        return (f"SubTask(task_id={self.task_id}, subtask_id={self.subtask_id}, "
                f"cpu_cycles={self.cpu_cycles}, remaining_data_size={self.data_size})")


# 任务类
class Task:
    def __init__(self, task_id, n_subtasks, vehicles):
        self.task_id = task_id
        self.n_subtasks = n_subtasks
        # 生成子任务，子任务的任务 ID 随机从车辆中选择
        self.subtasks = [SubTask(np.random.choice(list(vehicles)), i) for i in range(n_subtasks)]
        self.vehicles = vehicles
        self.dependencies = self.generate_custom_dag_dependencies()  # 生成任务依赖关系
        self.graph = self.build_dependency_graph()  # 构建任务依赖图
        self.is_completed = False  # 标志任务是否完成

    def generate_custom_dag_dependencies(self):
        """
        自定义生成任务间的依赖关系，使用 subtask_id 作为节点
        """
        dependencies = []

        # 左侧的串行部分 (例如：0 -> 1 -> 2)
        for i in range(self.n_subtasks // 2 - 1):
            dependencies.append((self.subtasks[i].subtask_id, self.subtasks[i + 1].subtask_id))

        # 中间解锁后可并行的任务 (例如：2 -> 3 和 2 -> 4)
        parallel_start = self.subtasks[self.n_subtasks // 2 - 1].subtask_id
        for i in range(self.n_subtasks // 2, self.n_subtasks - 1):
            dependencies.append((parallel_start, self.subtasks[i].subtask_id))

        # 修改逻辑：允许部分并行任务没有汇聚到最终节点
        final_task = self.subtasks[-1].subtask_id
        for i in range(self.n_subtasks // 2, self.n_subtasks - 1):
            if np.random.rand() > 0.5:
                dependencies.append((self.subtasks[i].subtask_id, final_task))

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

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Generated graph is not a DAG!")
        return G

    def function(self, subtask_id, process_amount):
        """
        处理指定子任务的数据，并动态更新任务状态
        :param subtask_id: 子任务ID
        :param process_amount: 本次处理的数据量
        """
        # 检查子任务是否存在
        if subtask_id not in self.graph:
            print(f"SubTask {subtask_id} does not exist in the task graph.")
            return

        # 处理子任务的数据
        subtask = self.subtasks[subtask_id]
        subtask.process_data(process_amount)
        print(f"Processed {process_amount} data for SubTask {subtask_id}: {subtask}")

        # 如果子任务完成，删除其所有出边
        if subtask.is_completed():
            successors = list(self.graph.successors(subtask_id))
            self.graph.remove_edges_from([(subtask_id, succ) for succ in successors])
            print(f"SubTask {subtask_id} completed. Removed outgoing edges: {successors}")

        # 检查所有叶子节点是否完成
        leaf_nodes = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]
        if all(self.subtasks[node].is_completed() for node in leaf_nodes):
            self.is_completed = True
            print(f"Task {self.task_id} is completed!")

    def plot_dependency_graph(self, color="lightblue"):
        """
        绘制任务依赖图，节点标签使用每个子任务的 task_id (即分配的车辆 ID)
        :param color: 节点颜色
        """
        pos = nx.spring_layout(self.graph)  # 使用 spring 布局

        # 节点标签为每个子任务的 task_id（分配的车辆 ID）
        labels = {node: f"{self.subtasks[node].task_id}" for node in self.graph.nodes}

        # 绘制图形
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_color=color, node_size=3000, font_size=10, font_weight="bold", font_color="black")

        plt.title(f"Dependency Graph for Task {self.task_id}")
        plt.show()

    def __repr__(self):
        return f"Task(task_id={self.task_id}, n_subtasks={self.n_subtasks}, is_completed={self.is_completed})"


