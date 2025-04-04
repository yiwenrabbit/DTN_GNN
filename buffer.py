import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, node_count, edge_dim, network_states_dim, x_dim, action_dim, ready_mask_dim, distance_mask_dim, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.node_count = node_count  # 固定节点数量
        self.edge_dim = edge_dim  # 固定边的数量（[num_edges, 2]）
        self.network_states_dim = network_states_dim
        self.x_dim = x_dim  # 节点特征维度
        self.action_dim = action_dim  # 动作维度
        self.batch_size = batch_size
        self.ready_mask_dim = ready_mask_dim
        self.distance_mask_dim = distance_mask_dim

        # 固定大小的存储
        self.x_memory = np.zeros((self.mem_size, self.node_count, self.x_dim))  # 节点特征
        self.edge_index_memory = np.zeros((self.mem_size, 2, self.edge_dim), dtype=int)  # 边索引

        self.new_x_memory = np.zeros((self.mem_size, self.node_count, self.x_dim))  # 下一状态的节点特征
        self.new_edge_index_memory = np.zeros((self.mem_size,  2, self.edge_dim,), dtype=int)  # 下一状态的边索引

        self.network_states_memory = np.zeros((self.mem_size, self.network_states_dim))  # 当前全局时延
        self.new_network_states_memory = np.zeros((self.mem_size, self.network_states_dim))  # 下一状态的全局时延

        self.ready_mask_memory = np.zeros((self.mem_size, self.ready_mask_dim))  # 当前 ready_mask
        self.new_ready_mask_memory = np.zeros((self.mem_size, self.ready_mask_dim))  # 下一状态的 ready_mask

        self.distance_mask_memory = np.zeros((self.mem_size, self.node_count, self.distance_mask_dim))  # 当前 distance_mask
        self.new_distance_mask_memory = np.zeros((self.mem_size, self.node_count, self.distance_mask_dim))  # 下一状态的 distance_mask

        self.done_mask_memory = np.zeros(
            (self.mem_size, self.ready_mask_dim))  # 当前 ready_mask
        self.new_done_mask_memory = np.zeros(
            (self.mem_size, self.ready_mask_dim))  # 下一状态的 ready_mask

        self.off_mask_memory = np.zeros(
            (self.mem_size, self.ready_mask_dim))  # 当前 off_mask
        self.new_off_mask_memory = np.zeros(
            (self.mem_size, self.ready_mask_dim))  # 下一状态的 off_mask

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.action_memory = np.zeros((self.mem_size, action_dim))  # 动作
        self.reward_memory = np.zeros(self.mem_size)  # 奖励
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)  # 终止标志

    def store_transition(self, x, edge_index, network_states, ready_mask, distance_mask, done_mask, off_mask, action, reward,
                         x_, edge_index_, new_network_states, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask, done):
        """
        存储一次转移样本
        """
        index = self.mem_cntr % self.mem_size

        # 存储当前状态
        self.x_memory[index] = x  # 节点特征
        self.edge_index_memory[index, :len(edge_index)] = edge_index  # 边索引
        self.network_states_memory[index] = network_states  # 当前网络状态
        self.ready_mask_memory[index] = ready_mask  # 当前 ready_mask
        self.distance_mask_memory[index] = distance_mask  # 当前 distance_mask
        self.done_mask_memory[index] = done_mask  # 当前 done_mask
        self.off_mask_memory[index] = off_mask  # 当前 off_mask



        # 存储下一状态
        self.new_x_memory[index] = x_  # 下一状态的节点特征
        self.new_edge_index_memory[index, :len(edge_index_)] = edge_index_  # 下一状态的边索引
        self.new_network_states_memory[index] = new_network_states  # 下一状态的网络状态
        self.new_ready_mask_memory[index] = new_ready_mask  # 下一状态的 ready_mask
        self.new_distance_mask_memory[index] = new_distance_mask  # 下一状态的 distance_mask
        self.new_done_mask_memory[index] = new_done_mask  # 当前 done_mask
        self.new_off_mask_memory[index] = new_off_mask  # 当前 done_mask

        # 存储动作、奖励和终止状态
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self):
        """
        从缓冲区中随机采样一个批次
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        # 随机选择一个批次的索引
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # 提取存储的数据
        x = self.x_memory[batch]  # 当前状态的节点特征
        edge_index = self.edge_index_memory[batch]  # 当前状态的边索引
        network_states = self.network_states_memory[batch]  # 当前全局时延
        ready_mask = self.ready_mask_memory[batch]  # 当前 ready_mask
        distance_mask = self.distance_mask_memory[batch]  # 当前 distance_mask
        done_mask = self.done_mask_memory[batch]       #当前子任务是否完成
        off_mask = self.off_mask_memory[batch]         #当前子任务是否在卸载

        actions = self.action_memory[batch]  # 动作
        rewards = self.reward_memory[batch]  # 奖励
        terminals = self.terminal_memory[batch]  # 终止标志

        new_x = self.new_x_memory[batch]  # 下一状态的节点特征
        new_edge_index = self.new_edge_index_memory[batch]  # 下一状态的边索引
        new_network_states = self.new_network_states_memory[batch]  # 下一状态的全局时延
        new_ready_mask = self.new_ready_mask_memory[batch]  # 下一状态的 ready_mask
        new_distance_mask = self.new_distance_mask_memory[batch]  # 下一状态的 distance_mask
        new_done_mask = self.new_done_mask_memory[batch]
        new_off_mask = self.new_off_mask_memory[batch]

        return x, edge_index, network_states, ready_mask, distance_mask, done_mask, off_mask, actions, rewards, new_x, new_edge_index, new_network_states, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask, terminals

    def ready(self):
        """
        判断缓冲区是否已准备好用于采样
        """
        return self.mem_cntr >= self.batch_size
