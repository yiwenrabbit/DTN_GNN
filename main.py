import agent
from environment import Env
import environment as ENVN
from agent import Agent
from buffer import ReplayBuffer
from GNN import GCNModel
import numpy as np
import torch as t
import copy
import os

def obs_list_to_state_vector(observation):
    obs_clone = observation.clone()
    state = np.array(obs_clone)
    return state

def clear_data(reward_dir, acc_dir, loss_dir):
    if os.path.exists(reward_dir):
        with open(reward_dir, 'w') as file:
            pass
    else:
        os.makedirs(os.path.dirname(reward_dir), exist_ok=True)

    if os.path.exists(acc_dir):
        with open(acc_dir, 'w') as file:
            pass
    else:
        os.makedirs(os.path.dirname(acc_dir), exist_ok=True)

    if os.path.exists(loss_dir):
        with open(loss_dir, 'w') as file:
            pass
    else:
        os.makedirs(os.path.dirname(loss_dir), exist_ok=True)

max_delay = ENVN.max_delay
num_class = agent.num_class

if __name__ == "__main__":
    subtask_num = ENVN.subtasks_dag
    max_size = 1000
    batch_size = 64
    edge_num = 16
    vehicle_num = 20

    reward_dir = "./data/reward"
    r_dir = ".txt"
    r_tmd = '_' + str(edge_num) + '_' + str(vehicle_num) + '_' + str(subtask_num)
    reward_dir = reward_dir + r_tmd + r_dir

    acc_dir = "./data/accuracy"
    acc_dir = acc_dir + r_tmd + r_dir

    loss_dir = "./data/loss"
    loss_dir = loss_dir + r_tmd + r_dir

    clear_data(reward_dir, acc_dir, loss_dir)

    ### GCN参数设定 ###
    Gcn_input_dim = 3
    Gcn_hidden_dim = 32
    Gcn_output_dim = 16
    Gcn_final_dim = num_class

    GCN_model = GCNModel(input_dim=Gcn_input_dim, hidden_dim=Gcn_hidden_dim, output_dim=Gcn_output_dim, num_classes=num_class)

    ### DRL参数设定 ###
    network_states_dim = subtask_num * (1 + 1 + edge_num + edge_num) + 1
    state_dim = subtask_num * Gcn_final_dim + network_states_dim
    n_actions = int(sum(np.ones(subtask_num) * (edge_num + 1)))

    ### 加载环境 ###
    env = Env.load_env()
    env_backup = copy.deepcopy(env)
    env.display_network()

    _, edges = env.get_gcn_inputs()
    edge_dim = edges.shape[1]
    distance_mask_dim = edge_num
    ready_mask_dim = subtask_num

    # 初始化 Agent 和 ReplayBuffer
    DT_place_agent = Agent(
        actor_dims=state_dim, critic_dims=state_dim, gcn_input_dim=Gcn_input_dim,
        gcn_hidden_dim=Gcn_hidden_dim, gcn_output_dim=Gcn_final_dim, n_actions=n_actions,
        agent_idx=0, n_subtasks=subtask_num, n_edges=edge_num,
        chkpt_dir='./tmp/', alpha=0.00001, beta=0.00001, fc1=256, fc2=128, fc3=64
    )

    buffer = ReplayBuffer(
        max_size=max_size, x_dim=Gcn_input_dim, network_states_dim=network_states_dim,
        node_count=subtask_num, edge_dim=edge_dim, action_dim=n_actions, batch_size=batch_size,
        ready_mask_dim=ready_mask_dim, distance_mask_dim=distance_mask_dim
    )

    ### 游戏开始 ###
    N_Games = 50
    MAX_STEPS = 100
    total_steps = 0
    best_score = -float('inf')
    all_ave_rewards = []
    all_task_accuracy = []
    all_loss = []

    for i in range(N_Games):
        if i > 0:
            env = copy.deepcopy(env_backup)

        score = 0
        done = False
        episode_step = 1
        print(f"\n===== Starting Episode {i + 1} =====")

        while not done:
            # 获取节点和边
            x, edges = env.get_gcn_inputs()
            network_state = env.get_network_state()
            ready_mask, distance_mask, done_mask, off_mask = env.generate_masks()

            # 确保特征维度匹配
            assert x.shape[1] == Gcn_input_dim, f"Mismatch in GCN input: {x.shape[1]} vs {Gcn_input_dim}"
            assert network_state.shape[0] == network_states_dim, f"Mismatch in network state: {network_state.shape[0]} vs {network_states_dim}"

            # 选择动作
            try:
                actions = DT_place_agent.choose_action(x, edges, network_state, total_steps, ready_mask, distance_mask, done_mask, off_mask)
                last_5_actions = actions[:,-5:]
                print(last_5_actions)
            except Exception as e:
                print(f"Error in choosing action: {e}")
                break

            # 环境 step
            try:
                reward, x_, edges_, new_network_state, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask = env.step(actions)
            except Exception as e:
                print(f"Error in environment step: {e}")
                break

            # 结束条件
            if episode_step >= MAX_STEPS or env.tasks[0].task_delay == 0 or env.tasks[0].is_completed:
                done = True
                ave_re = (score + reward) / episode_step
                all_task_accuracy.append(env.return_dt_accuracy())
                all_ave_rewards.append(ave_re)
                env.print_all_accuracy()
                if env.tasks[0].task_delay == 0 and not env.tasks[0].is_completed:
                    print(f"Task failed in Episode {i + 1}!")

            # 存储到 Replay Buffer
            buffer.store_transition(
                x, edges, network_state, ready_mask, distance_mask, done_mask, off_mask,
                actions, reward, x_, edges_, new_network_state, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask, done
            )

            # 学习
            try:
                critic_loss = DT_place_agent.learn(buffer, total_steps)
                all_loss.append(critic_loss)
            except Exception as e:
                print(f"Error in learning: {e}")
                break

            score += reward
            episode_step += 1
            total_steps += 1
            print(f"Episode {i + 1}, Step {episode_step}, Reward: {reward}, Score: {score}")

        # 保存模型
        if score > best_score:
            DT_place_agent.save_models()
            best_score = score

    # 保存结果
    with open(reward_dir, 'a') as file:
        file.write("\n".join(map(str, all_ave_rewards)) + '\n')

    with open(acc_dir, 'a') as file:
        file.write("\n".join(map(str, all_task_accuracy)) + '\n')

    with open(loss_dir, 'a') as file:
        file.write("\n".join(map(str, all_loss)) + '\n')
