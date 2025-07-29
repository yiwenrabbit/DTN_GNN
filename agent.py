# agent.py
import numpy as np
import torch as T
import torch.nn.functional as F
from Network import ActorNetwork, CriticNetwork
from GNN import GCNModel
from torch.nn.utils import clip_grad_norm_
import torch.distributions as dist


class PPOAgent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_subtasks, n_edges,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 alpha=0.0003, beta=0.0003, fc1=512, fc2=256, fc3=128, gamma=0.99,
                 gae_lambda=0.95, policy_clip=0.2, c1=0.5, c2=0.01, epochs=10):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.n_subtasks = n_subtasks
        self.n_edges = n_edges
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.c1 = c1  # value loss coefficient
        self.c2 = c2  # entropy coefficient
        self.epochs = epochs  # PPO训练轮数
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        # 添加权重衰减参数
        self.weight_decay = 1e-5

        # 学习率调度相关参数
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.lr_decay = 0.995

        # 初始化网络
        self.gcn = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim).to(self.device)
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, n_subtasks, n_edges,
                                  chkpt_dir=chkpt_dir, name=f'{self.agent_name}_actor').to(self.device)
        # PPO只需要一个Critic网络
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, 1,  # 注意这里改为输出单个值
                                    chkpt_dir=chkpt_dir, name=f'{self.agent_name}_critic').to(self.device)

        # 修改优化器，添加权重衰减
        self.gcn_optimizer = T.optim.AdamW(self.gcn.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.actor.optimizer = T.optim.AdamW(self.actor.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.critic.optimizer = T.optim.AdamW(self.critic.parameters(), lr=beta, weight_decay=self.weight_decay)

        # 添加学习率调度器
        self.actor_scheduler = T.optim.lr_scheduler.ExponentialLR(self.actor.optimizer, gamma=self.lr_decay)
        self.critic_scheduler = T.optim.lr_scheduler.ExponentialLR(self.critic.optimizer, gamma=self.lr_decay)
        self.gcn_scheduler = T.optim.lr_scheduler.ExponentialLR(self.gcn_optimizer, gamma=self.lr_decay)

        # 经验回放计数
        self.learn_step_counter = 0

    def check_and_fix_nan(self, tensor, name="tensor"):
        """修复NaN值"""
        if T.isnan(tensor).any():
            print(f"Warning: NaN detected in {name}")
            # 将NaN替换为0
            tensor = T.where(T.isnan(tensor), T.zeros_like(tensor), tensor)
        if T.isinf(tensor).any():
            print(f"Warning: Inf detected in {name}")
            # 将Inf替换为大值
            tensor = T.clamp(tensor, -1e6, 1e6)
        return tensor

    def choose_action(self, gcn_x, edge_index, network_states, episode,
                      ready_mask, distance_mask, done_mask, off_mask, explore=True):
        try:
            device = self.device

            def to_device(t, dtype=T.float):
                if isinstance(t, list):
                    t = T.tensor(t, dtype=dtype)
                elif isinstance(t, np.ndarray):
                    t = T.tensor(t, dtype=dtype)
                return t.to(device)

            gcn_x = to_device(gcn_x)
            edge_index = to_device(edge_index, dtype=T.long)
            ready_mask = to_device(ready_mask)
            distance_mask = to_device(distance_mask)
            done_mask = to_device(done_mask)
            off_mask = to_device(off_mask)
            network_states = to_device(network_states)

            # === 检查维度并扩展 ===
            if network_states.dim() == 1:
                network_states = network_states.unsqueeze(0)

            if gcn_x.dim() == 3:
                batch_size, num_nodes, _ = gcn_x.shape
            else:
                num_nodes = gcn_x.shape[0]
                batch_size = 1
            gcn_output = T.ones((batch_size, num_nodes, self.gcn.output_dim), device=device)
            gcn_flatten = gcn_output.view(batch_size, -1)

            # === 拼接完整状态向量 ===
            state = T.cat((network_states, gcn_flatten), dim=1)

            # === Actor前向传播 ===
            edge_logits, decision_probs = self.actor.forward(state, ready_mask, distance_mask, done_mask, off_mask)

            # === Edge 动作选择 ===
            batch_size, n_subtasks, n_edges = edge_logits.size()
            edge_actions = []
            edge_log_probs = []

            for i in range(n_subtasks):
                logits = edge_logits[0, i]
                logits = T.clamp(logits, -10, 10)
                probs = F.softmax(logits, dim=0)

                if T.isnan(probs).any():
                    probs = T.ones_like(probs) / len(probs)

                if explore:
                    m = dist.Categorical(probs)
                    action = m.sample()
                    log_prob = m.log_prob(action)
                else:
                    action = T.argmax(probs)
                    log_prob = T.log(probs[action] + 1e-8)

                edge_actions.append(action.item())
                edge_log_probs.append(log_prob)

            # === Decision 动作选择 ===
            decision_actions = decision_probs * ready_mask * off_mask
            decision_actions = T.clamp(decision_actions, 1e-8, 1 - 1e-8)
            decision_dist = dist.Beta(2.0, 2.0)
            decision_log_probs = decision_dist.log_prob(decision_actions)

            # === 拼接动作向量 ===
            edge_actions_tensor = T.tensor(edge_actions, dtype=T.long, device=device)
            edge_one_hot = F.one_hot(edge_actions_tensor, num_classes=n_edges).float()
            edge_flatten = edge_one_hot.view(1, -1)
            actions = T.cat([edge_flatten, decision_actions], dim=1)

            # === Critic 价值估计 ===
            value = self.critic.forward(state, T.zeros_like(actions))

            # === 返回值 ===
            total_log_prob = T.stack(edge_log_probs).sum() + decision_log_probs.sum()

            return actions.detach().cpu().numpy(), value.detach().cpu().numpy(), total_log_prob.detach().cpu().numpy()

        except Exception as e:
            print(f"[Choose Action Error] {e}")
            raise

    def compute_gae(self, rewards, values, dones, next_values):
        """计算广义优势估计(GAE)"""
        advantages = np.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_values
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def learn(self, memory, epoch):
        if not memory.ready():
            return

        self.learn_step_counter += 1

        try:
            # 从经验池中采样所有数据（PPO需要整个轨迹）
            (gcn_x, edge_index, network_states, ready_mask, distance_mask, done_mask, off_mask, actions,
             rewards, gcn_x_, edge_index_, new_network_states, new_ready_mask, new_distance_mask, new_done_mask,
             new_off_mask, dones) = memory.get_all_samples()

            # 数据转换到设备
            device = self.device
            gcn_x = T.tensor(gcn_x, dtype=T.float).to(device)
            edge_index = T.tensor(edge_index, dtype=T.long).to(device)
            network_states = T.tensor(network_states, dtype=T.float).to(device)
            ready_mask = T.tensor(ready_mask, dtype=T.float).to(device)
            distance_mask = T.tensor(distance_mask, dtype=T.float).to(device)
            done_mask = T.tensor(done_mask, dtype=T.float).to(device)
            off_mask = T.tensor(off_mask, dtype=T.float).to(device)
            actions = T.tensor(actions, dtype=T.float).to(device)
            rewards = T.tensor(rewards, dtype=T.float).to(device)
            dones = T.tensor(dones, dtype=T.bool).to(device)

            rewards = T.clamp(rewards, -100, 100)

            # === 计算旧的log概率和价值 ===
            with T.no_grad():
                gcn_output = self.gcn(gcn_x, edge_index)
                gcn_flatten = gcn_output.view(gcn_output.size(0), -1)
                states = T.cat((network_states, gcn_flatten), dim=1)

                old_values = self.critic.forward(states, T.zeros(states.size(0), self.n_actions).to(device)).flatten()

                # 计算旧的log概率
                edge_logits, decision_probs = self.actor.forward(states, ready_mask, distance_mask, done_mask, off_mask)

                # 分解动作
                edge_actions = actions[:, :self.n_subtasks * self.n_edges].view(-1, self.n_subtasks, self.n_edges)
                decision_actions = actions[:, self.n_subtasks * self.n_edges:]

                # 计算edge动作的log概率
                old_edge_log_probs = []
                for batch_idx in range(edge_actions.size(0)):
                    batch_log_probs = []
                    for subtask_idx in range(self.n_subtasks):
                        edge_action = edge_actions[batch_idx, subtask_idx].argmax()
                        logits = T.clamp(edge_logits[batch_idx, subtask_idx], -10, 10)
                        probs = F.softmax(logits, dim=0)
                        batch_log_probs.append(T.log(probs[edge_action] + 1e-8))
                    old_edge_log_probs.append(T.stack(batch_log_probs).sum())
                old_edge_log_probs = T.stack(old_edge_log_probs)

                # 计算决策动作的log概率
                decision_dist = dist.Beta(2.0, 2.0)
                old_decision_log_probs = decision_dist.log_prob(decision_actions.clamp(1e-8, 1 - 1e-8)).sum(dim=1)

                old_log_probs = old_edge_log_probs + old_decision_log_probs

            # === 计算优势和回报 ===
            advantages, returns = self.compute_gae(
                rewards.cpu().numpy(),
                old_values.cpu().numpy(),
                dones.cpu().numpy(),
                old_values[-1].cpu().numpy()
            )
            advantages = T.tensor(advantages, dtype=T.float).to(device)
            returns = T.tensor(returns, dtype=T.float).to(device)

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # === PPO训练循环 ===
            for _ in range(self.epochs):
                # 重新计算当前策略下的log概率和价值
                gcn_output = self.gcn(gcn_x, edge_index)
                gcn_flatten = gcn_output.view(gcn_output.size(0), -1)
                states = T.cat((network_states, gcn_flatten), dim=1)

                values = self.critic.forward(states, T.zeros(states.size(0), self.n_actions).to(device)).flatten()

                edge_logits, decision_probs = self.actor.forward(states, ready_mask, distance_mask, done_mask, off_mask)

                # 计算新的log概率
                new_edge_log_probs = []
                edge_entropy = []
                for batch_idx in range(edge_actions.size(0)):
                    batch_log_probs = []
                    batch_entropy = []
                    for subtask_idx in range(self.n_subtasks):
                        edge_action = edge_actions[batch_idx, subtask_idx].argmax()
                        logits = T.clamp(edge_logits[batch_idx, subtask_idx], -10, 10)
                        probs = F.softmax(logits, dim=0)
                        probs = probs + 1e-8
                        probs = probs / probs.sum()
                        batch_log_probs.append(T.log(probs[edge_action] + 1e-8))
                        batch_entropy.append(-(probs * T.log(probs + 1e-8)).sum())
                    new_edge_log_probs.append(T.stack(batch_log_probs).sum())
                    edge_entropy.append(T.stack(batch_entropy).sum())
                new_edge_log_probs = T.stack(new_edge_log_probs)
                edge_entropy = T.stack(edge_entropy).mean()

                decision_dist = dist.Beta(2.0, 2.0)
                new_decision_log_probs = decision_dist.log_prob(decision_actions.clamp(1e-8, 1 - 1e-8)).sum(dim=1)
                decision_entropy = decision_dist.entropy().mean()

                new_log_probs = new_edge_log_probs + new_decision_log_probs
                total_entropy = edge_entropy + decision_entropy

                # === 计算比率和损失 ===
                ratio = T.exp(new_log_probs - old_log_probs.detach())
                ratio = T.clamp(ratio, 0.1, 10)  # 防止比率过大

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
                policy_loss = -T.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns.detach())

                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * total_entropy

                if T.isnan(loss):
                    print(f"Warning: NaN loss get policy_loss: {policy_loss}, value_loss: {value_loss}")
                    return 0.0

                # 反向传播和优化
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.gcn_optimizer.zero_grad()

                loss.backward()

                # 梯度裁剪
                clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                clip_grad_norm_(self.gcn.parameters(), max_norm=0.5)

                self.actor.optimizer.step()
                self.critic.optimizer.step()
                self.gcn_optimizer.step()

            # 每1000次学习后更新学习率
            if self.learn_step_counter % 1000 == 0:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                self.gcn_scheduler.step()

            return loss.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in learning: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        T.save(self.gcn.state_dict(), f'{self.agent_name}_gcn.pth')

        # 保存优化器状态
        T.save({
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
            'gcn_optimizer': self.gcn_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'gcn_scheduler': self.gcn_scheduler.state_dict(),
        }, f'{self.agent_name}_optimizers.pth')

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        try:
            self.gcn.load_state_dict(T.load(f'{self.agent_name}_gcn.pth', map_location=self.device))

            # 加载优化器状态
            optimizers = T.load(f'{self.agent_name}_optimizers.pth', map_location=self.device)
            self.actor.optimizer.load_state_dict(optimizers['actor_optimizer'])
            self.critic.optimizer.load_state_dict(optimizers['critic_optimizer'])
            self.gcn_optimizer.load_state_dict(optimizers['gcn_optimizer'])
            self.actor_scheduler.load_state_dict(optimizers['actor_scheduler'])
            self.critic_scheduler.load_state_dict(optimizers['critic_scheduler'])
            self.gcn_scheduler.load_state_dict(optimizers['gcn_scheduler'])

        except Exception as e:
            print(f"Error loading models or optimizers: {e}")
