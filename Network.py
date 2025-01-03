import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.q = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


        self.to(self.device)

    def forward(self, state, action):
        action = action.reshape(state.shape[0], -1)  # 确保 action 的形状与 state 的批量一致
        input = T.cat([state, action], dim=1)  # 拼接状态和动作特征
        x = F.leaky_relu(self.fc1(input), 1)  # 直接输入到全连接层
        x = F.leaky_relu(self.fc2(x), 1)
        x = F.leaky_relu(self.fc3(x), 1)
        q = F.leaky_relu(self.q(x), 1)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims,
                 n_subtasks, n_edges, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.n_subtasks = n_subtasks
        self.n_edges = n_edges
        self.total_actions = n_subtasks * n_edges + n_subtasks  # Edge 动作 + 完成率动作

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # Fully connected layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)

        # Output layer for actions
        self.pi = nn.Linear(fc3_dims, self.total_actions)

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, ready_mask, distance_mask):
        """
        参数:
            state: 输入的状态特征 [batch_size, input_dims]
            distance_mask: [batch_size, n_subtasks, n_edges]，Edge 的可达掩码
            ready_mask: [batch_size, n_subtasks]，任务完成率的可用掩码
        返回:
            edge_probs: Edge 的选择概率 [batch_size, n_subtasks, n_edges]
            completion_rates: 任务完成率 [batch_size, n_subtasks]
        """
        # Forward pass through the network
        x = F.leaky_relu(self.fc1(state), negative_slope=0.1)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)

        # Raw actions output
        logits = self.pi(x)  # [batch_size, total_actions]

        # # 把mask变为tensor
        # distance_mask = T.tensor(distance_mask).to(self.device)  # 确保在设备上
        # ready_mask = T.tensor(ready_mask).to(self.device)  # 如果 ready_mask 也需要

        # Split logits into Edge logits and completion rate logits
        edge_logits = logits[:, :self.n_subtasks * self.n_edges]  # 前 n_subtasks * n_edges
        completion_logits = logits[:, self.n_subtasks * self.n_edges:]  # 后 n_subtasks

        # Reshape Edge logits to [batch_size, n_subtasks, n_edges]
        edge_logits = edge_logits.view(-1, self.n_subtasks, self.n_edges)

        # Apply distance mask to Edge logits
        masked_edge_logits = T.where(distance_mask > 0, edge_logits, T.tensor(-1e10).to(self.device))

        # Softmax over Edge logits (per subtask)
        edge_probs = F.softmax(masked_edge_logits, dim=2)  # [batch_size, n_subtasks, n_edges]

        # Apply tanh to completion logits and scale to [0, 1]
        raw_completion_rates = T.tanh(completion_logits)  # [-1, 1]
        completion_rates = (raw_completion_rates + 1) / 2  # [0, 1]

        # Apply ready mask to completion rates
        completion_rates = completion_rates * ready_mask  # [batch_size, n_subtasks]

        return edge_probs, completion_rates

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

