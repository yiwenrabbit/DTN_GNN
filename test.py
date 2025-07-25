import numpy as np
from agent import Agent
from buffer import ReplayBuffer

# 参数设置
state_dim = 10
action_dim = [3,4]
max_size = 1000
batch_size = 64

# 初始化 Agent 和 ReplayBuffer
agent = Agent(actor_dims=state_dim, critic_dims=state_dim, n_actions=action_dim, agent_idx=0,
              chkpt_dir='./tmp/', alpha=0.0001, beta=0.0001, fc1=64, fc2=64, fc3=64)
buffer = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=sum(action_dim), batch_size=batch_size)

# 模拟数据并填充经验回放
for _ in range(500):  # 填充 500 条经验
    state = np.random.rand(state_dim)
    action = np.random.rand(sum(action_dim))
    reward = np.random.rand()
    state_ = np.random.rand(state_dim)
    done = np.random.choice([False, True])

    buffer.store_transition(state, action, reward, state_, done)

# 开始训练
print("Starting training...")
for epoch in range(100):  # 训练 100 个 epoch
    if buffer.ready():
        agent.learn(buffer, epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Training step completed.")

print("Training completed.")
