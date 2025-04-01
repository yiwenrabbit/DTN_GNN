import numpy as np
import torch as T
import torch.nn.functional as F
from Network import ActorNetwork, CriticNetwork
from GNN import GCNModel

delta = 0.5  # Actor 损失权重



class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_subtasks, n_edges,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 alpha=0.0001, beta=0.0001, fc1=512, fc2=256, fc3=128, gamma=0.99, tau=0.005):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.n_subtasks = n_subtasks
        self.n_edges = n_edges
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        # 初始化网络
        self.gcn = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim).to(self.device)
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, n_subtasks, n_edges,
                                  chkpt_dir=chkpt_dir, name=f'{self.agent_name}_actor').to(self.device)
        self.critic1 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                     chkpt_dir=chkpt_dir, name=f'{self.agent_name}_critic1').to(self.device)
        self.critic2 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                     chkpt_dir=chkpt_dir, name=f'{self.agent_name}_critic2').to(self.device)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, n_subtasks, n_edges,
                                         chkpt_dir=chkpt_dir, name=f'{self.agent_name}_target_actor').to(self.device)
        self.target_critic1 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                            chkpt_dir=chkpt_dir, name=f'{self.agent_name}_target_critic1').to(self.device)
        self.target_critic2 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                            chkpt_dir=chkpt_dir, name=f'{self.agent_name}_target_critic2').to(self.device)

        # 优化器
        self.gcn_optimizer = T.optim.Adam(self.gcn.parameters(), lr=alpha)

        # 初始化目标网络
        self.update_network_parameters(tau=1)

    def choose_action(self, gcn_x, edge_index, network_states, episode, ready_mask, distance_mask, done_mask, off_mask,
                      alpha=1):
        # 动作探索的衰减策略
        alpha = max(0.05, alpha - 0.00005 * episode)
        temperature = max(0.4, 1.0 - 0.00005 * episode)  # Gumbel-Softmax 温度

        # 确保张量移动到正确的设备
        ready_mask = T.tensor(ready_mask, dtype=T.float).to(self.device)
        distance_mask = T.tensor(distance_mask, dtype=T.float).to(self.device)
        done_mask = T.tensor(done_mask, dtype=T.float).to(self.device)
        off_mask = T.tensor(off_mask, dtype=T.float).to(self.device)
        network_states = network_states.clone().detach().unsqueeze(0).to(self.device)

        # === GCN 前向传播 ===
        gcn_output = self.gcn(gcn_x, edge_index)  # GCN 输出
        gcn_flatten = gcn_output.flatten().unsqueeze(0)   # 展平 GCN 输出

        # === 拼接 network_states 和 GCN 输出 ===
        state = T.cat((network_states, gcn_flatten), dim=1)  # 拼接后的状态

        # === Actor 网络前向传播 ===
        edge_logits, decision_probs = self.actor.forward(state, ready_mask, distance_mask, done_mask, off_mask)

        # === Edge 动作选择（Gumbel-Softmax）===
        batch_size, n_subtasks, n_edges = edge_logits.size()
        edge_actions = []
        for logits in edge_logits[0]:  # 对每个子任务单独处理
            probs = F.softmax(logits / temperature, dim=0)
            sampled_edge = T.multinomial(probs, 1).item()
            edge_actions.append(sampled_edge)

        # === Decision 动作选择 ===
        decision_noise = alpha * (T.randn_like(decision_probs).to(self.device)+0.5)
        noisy_decision_probs = (decision_probs + decision_noise).clamp(0, 1)  # 加噪声
        decision_actions = noisy_decision_probs * ready_mask * off_mask

        # === 拼接完整动作 ===
        edge_probs_flatten = F.one_hot(T.tensor(edge_actions), num_classes=n_edges).float()
        actions = T.cat([edge_probs_flatten.view(1, -1), decision_actions], dim=1)

        return actions.clone().detach().cpu().numpy()

    def learn(self, memory, epoch):
        if not memory.ready():
            return
        T.autograd.set_detect_anomaly(True)

        # 从经验池中采样
        (gcn_x, edge_index, network_states, ready_mask, distance_mask, done_mask, off_mask, actions,
         rewards, gcn_x_, edge_index_, new_network_states, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask, dones) = memory.sample_buffer()

        # 数据转换到设备
        device = self.device
        gcn_x = T.tensor(gcn_x, dtype=T.float).to(device)
        edge_index = T.tensor(edge_index, dtype=T.long).to(device)
        network_states = T.tensor(network_states, dtype=T.float).to(device)
        ready_mask = T.tensor(ready_mask, dtype=T.float).to(device)
        distance_mask = T.tensor(distance_mask, dtype=T.float).to(device)
        done_mask = T.tensor(done_mask, dtype=T.float).to(device)
        off_mask = T.tensor(off_mask, dtype=T.float).to(device)

        new_ready_mask = T.tensor(new_ready_mask, dtype=T.float).to(device)
        new_distance_mask = T.tensor(new_distance_mask, dtype=T.float).to(device)
        new_done_mask = T.tensor(new_done_mask, dtype=T.float).to(device)
        new_off_mask = T.tensor(new_off_mask, dtype=T.float).to(device)


        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        gcn_x_ = T.tensor(gcn_x_, dtype=T.float).to(device)
        edge_index_ = T.tensor(edge_index_, dtype=T.long).to(device)
        new_network_states = T.tensor(new_network_states, dtype=T.float).to(device)
        dones = T.tensor(dones, dtype=T.bool).to(device)

        # === Critic 损失 ===
        gcn_output = self.gcn(gcn_x, edge_index)
        gcn_flatten = gcn_output.view(gcn_output.size(0), -1)
        states = T.cat((network_states, gcn_flatten), dim=1)

        gcn_output_ = self.gcn(gcn_x_, edge_index_).detach()
        gcn_flatten_ = gcn_output_.view(gcn_output.size(0), -1)   # 展平 GCN 输出
        states_ = T.cat((new_network_states, gcn_flatten_), dim=1)

        with T.no_grad():  # 目标网络不需要计算梯度
            edge_logits, decision_probs = self.target_actor.forward(states_, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask)

            # Edge 动作选择
            batch_size, n_subtasks, n_edges = edge_logits.size()
            target_edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(device)
            for batch_idx in range(batch_size):
                for subtask_idx in range(n_subtasks):
                    valid_logits = edge_logits[batch_idx, subtask_idx]
                    # 贪婪选择最大概率的 edge
                    target_edge_actions[batch_idx, subtask_idx] = valid_logits.argmax().item()



            # 完成率动作选择
            target_decision_probs = decision_probs * new_ready_mask * new_off_mask * new_done_mask

            # 将 edge 和 decision_probs 拼接为完整的 target actions
            edge_probs_flatten = F.one_hot(target_edge_actions, num_classes=n_edges).view(batch_size, -1).float()
            new_actions = T.cat([edge_probs_flatten, target_decision_probs], dim=1)

        # Critic 值计算
        critic1_value = self.critic1.forward(states, actions).flatten()
        critic2_value = self.critic2.forward(states, actions).flatten()
        target_critic1_value = self.target_critic1.forward(states_, new_actions).flatten()
        target_critic2_value = self.target_critic2.forward(states_, new_actions).flatten()

        # 最小目标值
        target_critic_value = T.min(target_critic1_value, target_critic2_value)
        target_critic_value[dones] = 0.0

        # Bellman 方程目标
        target = rewards + self.gamma * target_critic_value

        # Critic 损失
        critic1_loss = F.mse_loss(critic1_value, target.detach())
        critic2_loss = F.mse_loss(critic2_value, target.detach())
        critic_loss = critic1_loss + critic2_loss

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        self.gcn_optimizer.zero_grad()

        critic_loss.backward(retain_graph=True)
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()



        # === Actor 损失 ===
        edge_logits, decision_probs = self.actor.forward(states, ready_mask, distance_mask, done_mask, off_mask)
        actor_actions = T.cat([
            F.one_hot(edge_logits.argmax(dim=-1), num_classes=n_edges).view(batch_size, -1).float(),
            decision_probs
        ], dim=1)
        actor_loss = -self.critic1.forward(states, actor_actions).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # === GCN 损失 ===
        #self.gcn_optimizer.zero_grad()
        #gcn_loss = F.cross_entropy(gcn_output.view(-1, num_class), done_mask.view(-1).long())
        #gcn_loss.backward()
        self.gcn_optimizer.step()

        # 更新目标网络
        if epoch % 50 == 0:
            self.update_network_parameters()

        return critic_loss.detach().cpu().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_critic2.save_checkpoint()
        T.save(self.gcn.state_dict(), f'{self.agent_name}_gcn.pth')

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_critic2.load_checkpoint()
        self.gcn.load_state_dict(T.load(f'{self.agent_name}_gcn.pth'))
