import numpy as np
import torch as T
import torch.nn.functional as F
from Network import ActorNetwork, CriticNetwork
from GNN import GCNModel
from torch.nn.utils import clip_grad_norm_

# 修改参数
delta = 0.5  # Actor 损失权重


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_subtasks, n_edges,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 alpha=0.0003, beta=0.0003, fc1=512, fc2=256, fc3=128, gamma=0.99, tau=0.01):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.n_subtasks = n_subtasks
        self.n_edges = n_edges
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        # 添加权重衰减参数
        self.weight_decay = 1e-5

        # 学习率调度相关参数
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.lr_decay = 0.995  # 学习率衰减率

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
                                            chkpt_dir=chkpt_dir, name=f'{self.agent_name}_target_critic1').to(
            self.device)
        self.target_critic2 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                            chkpt_dir=chkpt_dir, name=f'{self.agent_name}_target_critic2').to(
            self.device)

        # 修改优化器，添加权重衰减
        self.gcn_optimizer = T.optim.AdamW(self.gcn.parameters(), lr=alpha, weight_decay=self.weight_decay)

        # 替换原有优化器
        self.actor.optimizer = T.optim.AdamW(self.actor.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.critic1.optimizer = T.optim.AdamW(self.critic1.parameters(), lr=beta, weight_decay=self.weight_decay)
        self.critic2.optimizer = T.optim.AdamW(self.critic2.parameters(), lr=beta, weight_decay=self.weight_decay)

        # 添加学习率调度器
        self.actor_scheduler = T.optim.lr_scheduler.ExponentialLR(self.actor.optimizer, gamma=self.lr_decay)
        self.critic1_scheduler = T.optim.lr_scheduler.ExponentialLR(self.critic1.optimizer, gamma=self.lr_decay)
        self.critic2_scheduler = T.optim.lr_scheduler.ExponentialLR(self.critic2.optimizer, gamma=self.lr_decay)
        self.gcn_scheduler = T.optim.lr_scheduler.ExponentialLR(self.gcn_optimizer, gamma=self.lr_decay)

        # 初始化目标网络
        self.update_network_parameters(tau=1)

        # 经验回放计数
        self.learn_step_counter = 0
        self.target_update_freq = 10  # 更频繁的目标网络更新

    def choose_action(self, gcn_x, edge_index, network_states, episode, ready_mask, distance_mask, done_mask, off_mask,
                      alpha=1):
        try:
            # 优化探索策略
            alpha = max(0.1, alpha - 0.00002 * episode)  # 更缓慢的噪声衰减
            temperature = max(0.5, 1.0 - 0.00002 * episode)  # 更缓慢的温度衰减

            # 确保所有输入都移动到相同的设备上
            gcn_x = T.tensor(gcn_x, dtype=T.float).to(self.device)
            edge_index = T.tensor(edge_index, dtype=T.long).to(self.device)
            ready_mask = T.tensor(ready_mask, dtype=T.float).to(self.device)
            distance_mask = T.tensor(distance_mask, dtype=T.float).to(self.device)
            done_mask = T.tensor(done_mask, dtype=T.float).to(self.device)
            off_mask = T.tensor(off_mask, dtype=T.float).to(self.device)

            # 确保 network_states 是 tensor 并且在正确的设备上
            if isinstance(network_states, np.ndarray):
                network_states = T.tensor(network_states, dtype=T.float)
            network_states = network_states.clone().detach().to(self.device)

            # 确保 network_states 是二维张量
            if network_states.dim() == 1:
                network_states = network_states.unsqueeze(0)

            # === GCN 前向传播 ===
            with T.no_grad():
                gcn_output = self.gcn(gcn_x, edge_index)  # GCN 输出

                # 确保 gcn_output 是二维张量，方便后续处理
                if gcn_output.dim() > 2:
                    gcn_flatten = gcn_output.flatten().unsqueeze(0)
                else:
                    gcn_flatten = gcn_output.flatten().unsqueeze(0)

                # === 拼接 network_states 和 GCN 输出 ===
                state = T.cat((network_states, gcn_flatten), dim=1)  # 拼接后的状态

                # === Actor 网络前向传播 ===
                edge_logits, decision_probs = self.actor.forward(state, ready_mask, distance_mask, done_mask, off_mask)

                # === Edge 动作选择（改进的Gumbel-Softmax）===
                batch_size, n_subtasks, n_edges = edge_logits.size()
                edge_actions = []

                # 采用epsilon-greedy策略来选择edge
                for logits in edge_logits[0]:  # 对每个子任务单独处理
                    if np.random.random() < alpha:  # 探索
                        probs = F.softmax(logits / temperature, dim=0)
                        sampled_edge = T.multinomial(probs, 1).item()
                    else:  # 利用
                        sampled_edge = logits.argmax().item()
                    edge_actions.append(sampled_edge)

                # === Decision 动作选择（改进的方法）===
                # 使用Beta分布噪声替代高斯噪声，更适合概率分布
                if np.random.random() < alpha:  # 探索
                    alpha_beta = 2.0  # Beta分布参数
                    beta_beta = 2.0  # Beta分布参数
                    decision_noise = alpha * T.distributions.Beta(alpha_beta, beta_beta).sample(
                        decision_probs.size()).to(self.device)
                    noisy_decision_probs = (decision_probs + decision_noise - 0.5).clamp(0, 1)  # 加噪声
                else:  # 利用
                    noisy_decision_probs = decision_probs

                decision_actions = noisy_decision_probs * ready_mask * off_mask

                # === 拼接完整动作 ===
                edge_actions_tensor = T.tensor(edge_actions, dtype=T.long).to(self.device)
                edge_probs_flatten = F.one_hot(edge_actions_tensor, num_classes=n_edges).float()
                if edge_probs_flatten.dim() == 2:
                    edge_probs_flatten = edge_probs_flatten.unsqueeze(0)

                edge_probs_flatten_view = edge_probs_flatten.view(1, -1)
                actions = T.cat([edge_probs_flatten_view, decision_actions], dim=1)

                return actions.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in choosing action: {e}")
            raise

    def learn(self, memory, epoch):
        if not memory.ready():
            return

        self.learn_step_counter += 1
        T.autograd.set_detect_anomaly(True)

        try:
            # 从经验池中采样
            (gcn_x, edge_index, network_states, ready_mask, distance_mask, done_mask, off_mask, actions,
             rewards, gcn_x_, edge_index_, new_network_states, new_ready_mask, new_distance_mask, new_done_mask,
             new_off_mask, dones) = memory.sample_buffer()

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

            # === GCN处理和状态组装 ===
            gcn_output = self.gcn(gcn_x, edge_index)
            gcn_flatten = gcn_output.view(gcn_output.size(0), -1)
            states = T.cat((network_states, gcn_flatten), dim=1)

            gcn_output_ = self.gcn(gcn_x_, edge_index_).detach()
            gcn_flatten_ = gcn_output_.view(gcn_output_.size(0), -1)
            states_ = T.cat((new_network_states, gcn_flatten_), dim=1)

            # === 目标Q值计算（使用带噪声的改进目标策略）===
            with T.no_grad():
                # 目标Actor动作生成
                edge_logits, decision_probs = self.target_actor.forward(states_, new_ready_mask, new_distance_mask,
                                                                        new_done_mask, new_off_mask)

                # 添加目标策略噪声 (TD3算法特性)
                noise_std = 0.2  # 噪声标准差
                noise_clip = 0.5  # 噪声裁剪
                # Edge动作处理
                batch_size, n_subtasks, n_edges = edge_logits.size()
                target_edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(device)
                for batch_idx in range(batch_size):
                    for subtask_idx in range(n_subtasks):
                        # 添加噪声到logits
                        noise = T.randn_like(edge_logits[batch_idx, subtask_idx]) * noise_std
                        noise = T.clamp(noise, -noise_clip, noise_clip)
                        noisy_logits = edge_logits[batch_idx, subtask_idx] + noise
                        target_edge_actions[batch_idx, subtask_idx] = noisy_logits.argmax().item()

                # 决策值处理
                noise = T.randn_like(decision_probs) * noise_std
                noise = T.clamp(noise, -noise_clip, noise_clip)
                noisy_decision_probs = (decision_probs + noise).clamp(0, 1)
                target_decision_probs = noisy_decision_probs * new_ready_mask * new_off_mask * new_done_mask

                # 拼接完整目标动作
                edge_probs_flatten = F.one_hot(target_edge_actions, num_classes=n_edges).view(batch_size, -1).float()
                target_actions = T.cat([edge_probs_flatten, target_decision_probs], dim=1)

                # 目标Q值计算
                target_critic1_value = self.target_critic1.forward(states_, target_actions).flatten()
                target_critic2_value = self.target_critic2.forward(states_, target_actions).flatten()
                target_critic_value = T.min(target_critic1_value, target_critic2_value)

                # 改进的Bellman目标计算
                target = rewards + self.gamma * (1 - dones.float()) * target_critic_value

            # === Critic 网络更新 ===
            # 当前Q值
            current_critic1_value = self.critic1.forward(states, actions).flatten()
            current_critic2_value = self.critic2.forward(states, actions).flatten()

            # 计算Huber损失（比MSE更稳定）
            critic1_loss = F.smooth_l1_loss(current_critic1_value, target.detach())
            critic2_loss = F.smooth_l1_loss(current_critic2_value, target.detach())
            critic_loss = critic1_loss + critic2_loss

            # 清零梯度
            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            self.gcn_optimizer.zero_grad()

            # 反向传播
            critic_loss.backward(retain_graph=True)

            # 梯度裁剪
            clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            clip_grad_norm_(self.gcn.parameters(), max_norm=1.0)

            # 优化器步进
            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            # === Actor 网络更新（减少更新频率，提高稳定性）===
            if self.learn_step_counter % 10 == 0:  # 每两步更新一次Actor
                # 前向传播生成动作
                edge_logits, decision_probs = self.actor.forward(states, ready_mask, distance_mask, done_mask, off_mask)
                # One-hot编码edge动作
                actor_edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(device)
                for batch_idx in range(batch_size):
                    for subtask_idx in range(n_subtasks):
                        actor_edge_actions[batch_idx, subtask_idx] = edge_logits[batch_idx, subtask_idx].argmax().item()

                edge_probs_flatten = F.one_hot(actor_edge_actions, num_classes=n_edges).view(batch_size, -1).float()

                # 处理决策值
                actor_decision_probs = decision_probs * ready_mask * off_mask

                # 拼接完整Actor动作
                actor_actions = T.cat([edge_probs_flatten, actor_decision_probs], dim=1)

                # 计算策略梯度损失
                actor_loss = -self.critic1.forward(states, actor_actions).mean()

                # 添加熵正则化项，促进探索
                entropy_reg = 0.01  # 熵正则化系数
                edge_entropy = -T.sum(F.softmax(edge_logits, dim=-1) * F.log_softmax(edge_logits, dim=-1))
                decision_entropy = -T.sum(decision_probs * T.log(decision_probs + 1e-10) +
                                          (1 - decision_probs) * T.log(1 - decision_probs + 1e-10))
                entropy = edge_entropy + decision_entropy
                actor_loss = actor_loss - entropy_reg * entropy  # 鼓励探索

                # 梯度更新
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor.optimizer.step()

            # GCN优化
            self.gcn_optimizer.step()

            # 更新目标网络
            if self.learn_step_counter % self.target_update_freq == 0:
                self.update_network_parameters()

            # 每1000次学习后更新学习率
            if self.learn_step_counter % 1000 == 0:
                self.actor_scheduler.step()
                self.critic1_scheduler.step()
                self.critic2_scheduler.step()
                self.gcn_scheduler.step()

                # 输出当前学习率
                for param_group in self.actor.optimizer.param_groups:
                    current_lr = param_group['lr']
                    print(f"Current actor learning rate: {current_lr}")

            return critic_loss.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in learning: {e}")
            import traceback
            traceback.print_exc()
            raise

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

        # 保存优化器状态
        T.save({
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic1_optimizer': self.critic1.optimizer.state_dict(),
            'critic2_optimizer': self.critic2.optimizer.state_dict(),
            'gcn_optimizer': self.gcn_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic1_scheduler': self.critic1_scheduler.state_dict(),
            'critic2_scheduler': self.critic2_scheduler.state_dict(),
            'gcn_scheduler': self.gcn_scheduler.state_dict(),
        }, f'{self.agent_name}_optimizers.pth')

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_critic2.load_checkpoint()
        try:
            self.gcn.load_state_dict(T.load(f'{self.agent_name}_gcn.pth', map_location=self.device))

            # 加载优化器状态
            optimizers = T.load(f'{self.agent_name}_optimizers.pth', map_location=self.device)
            self.actor.optimizer.load_state_dict(optimizers['actor_optimizer'])
            self.critic1.optimizer.load_state_dict(optimizers['critic1_optimizer'])
            self.critic2.optimizer.load_state_dict(optimizers['critic2_optimizer'])
            self.gcn_optimizer.load_state_dict(optimizers['gcn_optimizer'])
            self.actor_scheduler.load_state_dict(optimizers['actor_scheduler'])
            self.critic1_scheduler.load_state_dict(optimizers['critic1_scheduler'])
            self.critic2_scheduler.load_state_dict(optimizers['critic2_scheduler'])
            self.gcn_scheduler.load_state_dict(optimizers['gcn_scheduler'])

        except Exception as e:
            print(f"Error loading models or optimizers: {e}")