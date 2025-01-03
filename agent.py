import numpy as np
import torch as T
from Network import ActorNetwork, CriticNetwork
import torch.nn.functional as F
from GNN import GCNModel


def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** (epoch // 5000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_subtasks, n_edges,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 alpha=0.0001, beta=0.0001, fc1=512, fc2=256, fc3=128, gamma=0.99, tau=0.001):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.n_subtasks = n_subtasks
        self.n_edges = n_edges
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx

        # 初始化网络
        self.gcn = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim).to(self.device)
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, n_subtasks, n_edges,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor').to(self.device)
        self.critic1 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                     chkpt_dir=chkpt_dir, name=self.agent_name + '_critic1').to(self.device)
        self.critic2 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                     chkpt_dir=chkpt_dir, name=self.agent_name + '_critic2').to(self.device)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, n_subtasks, n_edges,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor').to(self.device)
        self.target_critic1 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                            chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic1').to(
            self.device)
        self.target_critic2 = CriticNetwork(beta, critic_dims, fc1, fc2, fc3, n_actions,
                                            chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic2').to(
            self.device)

        # 优化器
        self.gcn_optimizer = T.optim.Adam(self.gcn.parameters(), lr=alpha)

        self.update_network_parameters(tau=1)

    def choose_action(self, gcn_x, edge_index, global_delay, episode, ready_mask, distance_mask, alpha=0.1):
        # 确保 distance_mask 是一个张量
        if not isinstance(distance_mask, T.Tensor):
            distance_mask = T.tensor(distance_mask, dtype=T.float)

        # 确保 ready_mask 是一个张量
        if not isinstance(ready_mask, T.Tensor):
            ready_mask = T.tensor(ready_mask, dtype=T.float)

        # 将张量移动到设备上
        distance_mask = distance_mask.to(self.device)
        ready_mask = ready_mask.to(self.device)

        # GCN 前向传播
        gcn_output = self.gcn(gcn_x, edge_index)
        gcn_flatten = gcn_output.flatten().unsqueeze(0)

        # 拼接全局时延特征
        global_delay_tensor = T.tensor([global_delay], dtype=T.float).to(self.device)
        global_delay_tensor = global_delay_tensor.unsqueeze(1)
        state = T.cat((global_delay_tensor, gcn_flatten), dim=1)

        # Actor 网络前向传播
        edge_logits, completion_logits = self.actor.forward(state, ready_mask, distance_mask)

        # Edge 动作选择
        batch_size, n_subtasks, n_edges = edge_logits.size()
        edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(self.device)
        for batch_idx in range(batch_size):
            for subtask_idx in range(n_subtasks):
                valid_logits = edge_logits[batch_idx, subtask_idx]
                if T.rand(1).item() < alpha:
                    valid_edges = T.arange(n_edges).to(self.device)
                    random_edge = T.randint(len(valid_edges), (1,)).item()
                    edge_actions[batch_idx, subtask_idx] = valid_edges[random_edge]
                else:
                    edge_actions[batch_idx, subtask_idx] = valid_logits.argmax().item()

        # 完成率动作选择
        completion_noise = 0.01 * T.randn_like(completion_logits).to(self.device) * (0.1 ** (episode // 10000))
        completion_probs = completion_logits + completion_noise
        completion_probs = (completion_probs * ready_mask).clamp(0, 1)

        # 拼接动作为完整的输出
        edge_probs_flatten = F.one_hot(edge_actions, num_classes=n_edges).view(batch_size, -1).float()
        actions = T.cat([edge_probs_flatten, completion_probs], dim=1)

        return actions.detach().cpu().numpy()

    def learn(self, memory, epoch):
        if not memory.ready():
            return

        # 从经验池中采样
        (gcn_x, edge_index, global_delay, ready_mask, distance_mask, actions,
         rewards, gcn_x_, edge_index_, global_delay_, new_ready_mask, new_distance_mask, dones) = memory.sample_buffer()

        # 确保 distance_mask 是一个张量
        if not isinstance(distance_mask, T.Tensor):
            distance_mask = T.tensor(distance_mask, dtype=T.float)

        # 确保 ready_mask 是一个张量
        if not isinstance(ready_mask, T.Tensor):
            ready_mask = T.tensor(ready_mask, dtype=T.float)

        # 确保 new_distance_mask 是一个张量
        if not isinstance(new_distance_mask, T.Tensor):
            new_distance_mask = T.tensor(new_distance_mask, dtype=T.float)

        # 确保 new_ready_mask 是一个张量
        if not isinstance(new_ready_mask, T.Tensor):
            new_ready_mask = T.tensor(new_ready_mask, dtype=T.float)


        device = self.device
        gcn_x = T.tensor(gcn_x, dtype=T.float).to(device)
        edge_index = T.tensor(edge_index, dtype=T.long).to(device)
        global_delay = T.tensor(global_delay, dtype=T.float).to(device)

        distance_mask = distance_mask.to(device)
        ready_mask = ready_mask.to(device)

        new_distance_mask = new_distance_mask.to(device)
        new_ready_mask = new_ready_mask.to(device)

        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        gcn_x_ = T.tensor(gcn_x_, dtype=T.float).to(device)
        edge_index_ = T.tensor(edge_index_, dtype=T.long).to(device)
        global_delay_ = T.tensor(global_delay_, dtype=T.float).to(device)


        dones = T.tensor(dones, dtype=T.bool).to(device)


        # GCN 前向传播
        gcn_output = self.gcn(gcn_x, edge_index)
        # 展平 GCN 输出
        gcn_flatten = gcn_output.view(gcn_output.size(0), -1)
        # 拼接 global_delay
        global_delay_cloned = global_delay.clone().detach()
        states = T.cat((global_delay_cloned.unsqueeze(1), gcn_flatten), dim=1)

        #同理处理新的状态
        gcn_output_ = self.gcn(gcn_x_, edge_index_)
        gcn_new_flatten = gcn_output_.view(gcn_output.size(0), -1)
        new_global_delay_cloned = global_delay_.clone().detach()
        states_ = T.cat((new_global_delay_cloned.unsqueeze(1), gcn_new_flatten), dim=1)

        # Compute target actions
        with T.no_grad():  # 目标网络不需要计算梯度
            edge_logits, completion_logits = self.target_actor.forward(states_, new_ready_mask, new_distance_mask)

            # Edge 动作选择
            batch_size, n_subtasks, n_edges = edge_logits.size()
            target_edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(device)
            for batch_idx in range(batch_size):
                for subtask_idx in range(n_subtasks):
                    valid_logits = edge_logits[batch_idx, subtask_idx]
                    # 贪婪选择最大概率的 edge
                    target_edge_actions[batch_idx, subtask_idx] = valid_logits.argmax().item()

            # 完成率动作选择
            completion_noise = 0.01 * T.randn_like(completion_logits).to(device) * (0.1 ** (epoch // 10000))
            target_completion_probs = completion_logits + completion_noise
            target_completion_probs = (target_completion_probs * new_ready_mask).clamp(0, 1)

            # 将 edge 和 completion 拼接为完整的 target actions
            edge_probs_flatten = F.one_hot(target_edge_actions, num_classes=n_edges).view(batch_size, -1).float()
            new_actions = T.cat([edge_probs_flatten, target_completion_probs], dim=1)

            # # 确保完整打印张量内容
            # T.set_printoptions(threshold=10000, edgeitems=10, linewidth=1000, sci_mode=False)
            #
            # # 打印 new_actions 的值
            # print("new_actions:", new_actions)
            # print('pass')

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

        #测试是否有梯度
        # print('critic反向传播前')
        # for name, param in self.gcn.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: Gradient exists")
        #     else:
        #         print(f"{name}: No gradient")

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # 测试梯度是否关联
        # gcn_grads = T.autograd.grad(critic_loss, self.gcn.parameters(), retain_graph=True)
        # print(gcn_grads)  # 如果为 None，说明梯度未传递
        # 测试是否有梯度
        # print('critic反向传播后')
        # for name, param in self.gcn.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: Gradient exists")
        #     else:
        #         print(f"{name}: No gradient")

        # Actor 损失################首先计算当前的action
        # 获取 actor 的输出：edge_logits 和 completion_logits
        edge_logits, completion_logits = self.actor.forward(states, ready_mask, distance_mask)

        # Edge 动作选择
        batch_size, n_subtasks, n_edges = edge_logits.size()
        edge_actions = T.zeros(batch_size, n_subtasks, dtype=T.long).to(self.device)

        for batch_idx in range(batch_size):
            for subtask_idx in range(n_subtasks):
                valid_logits = edge_logits[batch_idx, subtask_idx]
                edge_actions[batch_idx, subtask_idx] = valid_logits.argmax().item()

        # 完成率动作选择
        completion_probs = T.tanh(completion_logits)  # 使用 tanh 确保值在 [-1, 1] 之间
        completion_probs = (completion_probs * ready_mask).clamp(0, 1)  # 应用掩码

        # 将 edge 和 completion 拼接为完整的动作
        edge_probs_flatten = F.one_hot(edge_actions, num_classes=n_edges).view(batch_size, -1).float()
        actor_actions = T.cat([edge_probs_flatten, completion_probs], dim=1)
        # 计算 Actor 损失
        actor_loss = -self.critic1.forward(states, actor_actions).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()



        # GCN 损失（可选）之前critic和actor的反向传播已经计算了梯度，直接更新即可
        self.gcn_optimizer.step()

        # 更新目标网络
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_critic2.save_checkpoint()
        self.gcn.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_critic2.load_checkpoint()
        self.gcn.load_checkpoint()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)  # 对目标动作网络的更新

        target_critic_params = self.target_critic1.named_parameters()
        critic_params = self.critic1.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic1.load_state_dict(critic_state_dict)  # 对目标价值网络的更新

        target_critic2_params = self.target_critic2.named_parameters()
        critic2_params = self.critic2.named_parameters()

        target_critic2_state_dict = dict(target_critic2_params)
        critic2_state_dict = dict(critic2_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic2_state_dict[name].clone() + \
                                      (1 - tau) * target_critic2_state_dict[name].clone()

        self.target_critic2.load_state_dict(critic_state_dict)
