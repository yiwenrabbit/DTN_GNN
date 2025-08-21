# main.py
import agent
from environment import Env
import environment as ENVN
from agent import PPOAgent  # 改为导入PPOAgent
from buffer import ReplayBuffer
from GNN import GCNModel
import numpy as np
import torch as t
import copy
import os
import matplotlib.pyplot as plt
import datetime
import wandb

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

wandb.login(key="122149ff6074b3fffbfa05229528d9cfa0a6835b")
wandb.init(
    project="DTN-PPO",
    name=f"{current_time}"
)


def reset_reward(ave_acc, transitions):
    return 0


def obs_list_to_state_vector(observation):
    obs_clone = observation.clone()
    state = np.array(obs_clone)
    return state


dic = {}


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


def process_accuracy(accuracy_data):
    """处理准确率数据，确保返回float类型"""
    if isinstance(accuracy_data, (list, tuple)):
        # 如果是列表或元组，返回平均值
        return float(np.mean(accuracy_data))
    elif isinstance(accuracy_data, (int, float)):
        return float(accuracy_data)
    else:
        # 如果是其他类型，尝试转换为float
        try:
            return float(accuracy_data)
        except:
            return 0.0


max_delay = ENVN.max_delay

if __name__ == "__main__":
    subtask_num = ENVN.subtasks_dag
    max_size = 1000
    batch_size = 64
    edge_num = 16
    vehicle_num = 20

    reward_dir = "./data/reward_ppo"
    r_dir = ".txt"
    r_tmd = '_' + str(edge_num) + '_' + str(vehicle_num) + '_' + str(subtask_num)
    reward_dir = reward_dir + r_tmd + r_dir

    acc_dir = "./data/accuracy_ppo"
    acc_dir = acc_dir + r_tmd + r_dir

    loss_dir = "./data/loss_ppo"
    loss_dir = loss_dir + r_tmd + r_dir

    clear_data(reward_dir, acc_dir, loss_dir)

    ### GCN参数设定 ###
    Gcn_input_dim = 3
    Gcn_hidden_dim = 128
    Gcn_output_dim = 128

    ### DRL参数设定 ###
    network_states_dim = subtask_num * (1 + 1 + edge_num + edge_num) + 1
    state_dim = subtask_num * Gcn_output_dim + network_states_dim
    n_actions = int(sum(np.ones(subtask_num) * (edge_num + 1)))

    ### 加载环境 ###
    env = Env.load_env()

    if env is None:
        # 环境不存在，自动重新创建一个环境实例并保存
        env = Env(edge_num, vehicle_num, False)
        env.save_env()

    env_backup = copy.deepcopy(env)
    env.display_network()
    env.tasks[0].plot_dependency_graph()

    _, edges = env.get_gcn_inputs()
    edge_dim = edges.shape[1]
    distance_mask_dim = edge_num
    ready_mask_dim = subtask_num

    # 初始化 PPO Agent 和 ReplayBuffer
    DT_place_agent = PPOAgent(
        actor_dims=state_dim, critic_dims=state_dim, gcn_input_dim=Gcn_input_dim,
        gcn_hidden_dim=Gcn_hidden_dim, gcn_output_dim=Gcn_output_dim, n_actions=n_actions,
        agent_idx=0, n_subtasks=subtask_num, n_edges=edge_num,
        chkpt_dir='./tmp/', alpha=0.0003, beta=0.0003, fc1=256, fc2=128, fc3=64,
        gamma=0.99, gae_lambda=0.95, policy_clip=0.2, c1=0.5, c2=0.01, epochs=10
    )

    buffer = ReplayBuffer(
        max_size=max_size, x_dim=Gcn_input_dim, network_states_dim=network_states_dim,
        node_count=subtask_num, edge_dim=edge_dim, action_dim=n_actions, batch_size=batch_size,
        ready_mask_dim=ready_mask_dim, distance_mask_dim=distance_mask_dim
    )

    ### 游戏开始 ###
    N_Games = 50000
    MAX_STEPS = 200
    total_steps = 0
    best_score = -float('inf')
    all_ave_rewards = []
    all_task_accuracy = []
    all_loss = []
    # all_score = []
    episode_rewards = []  # 每局分数：完成=改写后的总分；未完成/超时=原始累计score
    episode_completed_flags = []  # 每局是否完成(1/0)，可选

    # task_spread = env.tasks[0].calculate_task_spread()
    task_status_map = {}

    # PPO特有的变量
    episode_values = []
    episode_log_probs = []

    for i in range(N_Games):
        if i > 0:
            env = copy.deepcopy(env_backup)

        # 重置episode相关变量
        episode_values = []
        episode_log_probs = []
        last_state = 0  # 上一个时隙在更新表示为0， 1表示上个时隙就是在卸载可以跳过
        temp_x, temp_edges, temp_network_state, temp_ready_mask, temp_distance_mask, temp_done_mask, temp_off_mask, temp_actions, temp_reward = None, None, None, None, None, None, None, None, None,
        next_state = 0  # 下一个需要记录的状态来了，0表示没来，1表示来了
        score = 0
        done = False
        episode_step = 1
        print(f"\n===== Starting Episode {i + 1} =====")

        episode_ave_accuracy = 0  # 记录当前episode的平均奖励
        episode_transitions = []  # 用于缓存一个episode的所有step

        while not done:
            # 获取节点和边
            x, edges = env.get_gcn_inputs()
            network_state = env.get_network_state()
            ready_mask, distance_mask, done_mask, off_mask = env.generate_masks()

            # 确保特征维度匹配
            assert x.shape[1] == Gcn_input_dim, f"Mismatch in GCN input: {x.shape[1]} vs {Gcn_input_dim}"
            assert network_state.shape[
                       0] == network_states_dim, f"Mismatch in network state: {network_state.shape[0]} vs {network_states_dim}"

            # 选择动作
            try:
                actions, value, log_prob = DT_place_agent.choose_action(
                    x, edges, network_state, i, ready_mask, distance_mask,
                    done_mask, off_mask, explore=True
                )
                episode_values.append(value)
                episode_log_probs.append(log_prob)
                last_5_actions = actions[:, -5:]
                print(f"Step {episode_step}, Last 5 actions: {last_5_actions}")
            except Exception as e:
                print(f"Error in choosing action: {e}")
                break

            # 环境 step
            try:
                reward, x_, edges_, new_network_state, new_ready_mask, new_distance_mask, new_done_mask, new_off_mask, update_ratio_all = env.step(
                    actions)
            except Exception as e:
                print(f"Error in environment step: {e}")
                break

            for count in env.tasks:
                print("delay", count.task_delay)

            # 结束条件
            if episode_step >= MAX_STEPS or env.tasks[0].task_delay == 0 or env.tasks[0].is_completed:
                done = True
                ave_re = (score + reward) / episode_step
                min_accuracy, ave_accurarcy = env.return_dt_accuracy()
                # processed_accuracy = process_accuracy(raw_accuracy)
                episode_ave_accuracy = ave_accurarcy
                all_task_accuracy.append(ave_accurarcy)
                # all_score.append(score + reward)
                all_ave_rewards.append(ave_re)
                task_status = env.print_all_accuracy()
                wandb.log({
                    f"Task_1_status": task_status[0],
                    f"Task_2_status": task_status[1],
                    f"Task_3_status": task_status[2],
                    f"Task_4_status": task_status[3],
                    f"Task_5_status": task_status[4],
                }, step=i + 1)

                if env.tasks[0].task_delay == 0 and not env.tasks[0].is_completed:
                    print(f"Task failed in Episode {i + 1}!")
                # env.apply_precision_gap_penalty(buffer)

            episode_transitions.append([x, edges, network_state, ready_mask, distance_mask, done_mask, off_mask,
                                        actions, reward, x_, edges_, new_network_state, new_ready_mask,
                                        new_distance_mask, new_done_mask,
                                        new_off_mask, done, update_ratio_all])

            if not done:
                score += reward
                episode_step += 1
                total_steps += 1
                print(f"Episode {i + 1}, Step {episode_step}, Reward: {reward:.4f}, Score: {score:.4f}")

        # ========== 一局游戏结束后（统一按完成/未完成两种情况记分） ==========
        task_completed = env.tasks[0].is_completed

        if task_completed:
            # 1) 关键步 + 惩罚字典 + 改写奖励
            relevant_steps, filted_transitions = env.find_unique_relevant_steps(episode_transitions)
            updated_steps_dict = env.return_update_steps(relevant_steps)
            delay_penalty_dict = env.return_delay_penalty(updated_steps_dict)

            accuracy_step_dict = {
                subtask_id: (env.subtask_accuracy_dict[subtask_id], env.last_update_step[subtask_id])
                for subtask_id in env.subtask_accuracy_dict
                if subtask_id in env.last_update_step
            }

            updated_transitions = env.update_rewards(
                filted_transitions, delay_penalty_dict, accuracy_step_dict, relevant_steps, episode_ave_accuracy
            )

            # 2) 本局最终得分（完成时）= 改写后的奖励总和（transition[8]）
            episode_score_post = sum(tr[8] for tr in updated_transitions)

            # 3) 存入 buffer（完成局用改写后的 transitions）
            for transition in updated_transitions:
                (x, edges, network_state, ready_mask, distance_mask, done_mask, off_mask,
                 actions, reward, x_, edges_, new_network_state, new_ready_mask, new_distance_mask, new_done_mask,
                 new_off_mask, done, _) = transition

                buffer.store_transition(
                    x, edges, network_state, ready_mask, distance_mask, done_mask, off_mask,
                    actions, reward, x_, edges_, new_network_state, new_ready_mask,
                    new_distance_mask, new_done_mask, new_off_mask, done
                )

            # 4) 学习
            if buffer.ready():
                try:
                    print(f"Learning from episode {i + 1}...")
                    loss = DT_place_agent.learn(buffer, i)
                    all_loss.append(loss)
                    print(f"Loss: {loss:.4f}")
                except Exception as e:
                    print(f"Error in learning: {e}")
                buffer.clear()

            # 5) 更新“每局分数”和完成标记
            episode_rewards.append(episode_score_post)
            episode_completed_flags.append(1)

            # 6) wandb 记录（完成）
            min_acc, avg_acc = env.return_dt_accuracy()
            wandb.log({
                "Episode_Reward": episode_score_post,  # 完成局：用改写后分数
                "Episode_Reward_raw": float(score),  # 也记录原始累计（对比用）
                "Episode_Completed": 1,
                "Delay": env.tasks[0].task_delay,
                "Episode_Steps": episode_step,
                "Episode_MinAcc": float(min_acc),
                "Episode_AvgAcc": float(avg_acc),
                "Total_Steps": total_steps
            }, step=i + 1)

            # 7) 保存最优模型（按“每局分数”评估）
            if episode_score_post > best_score:
                print(f"New best score: {episode_score_post:.4f} (previous: {best_score:.4f})")
                DT_place_agent.save_models()
                best_score = episode_score_post

        else:
            # 未完成/超时：这局也要记 —— 用“原始累计分数 score”作为该局分数
            episode_rewards.append(score)
            episode_completed_flags.append(0)

            # wandb 记录（未完成）
            wandb.log({
                "Episode_Reward": float(score),  # 未完成局：用原始累计分数
                "Episode_Completed": 0,
                "Delay": env.tasks[0].task_delay,
                "Episode_Steps": episode_step,
                "Total_Steps": total_steps
            }, step=i + 1)

        # 定期输出训练进度
        if (i + 1) % 10 == 0:
            recent = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            avg_recent = np.mean(recent) if recent else 0.0
            print(f"\n=== Training Progress ===")
            print(f"Episodes completed: {i + 1}/{N_Games}")
            print(f"Average score (last 10 episodes): {avg_recent:.4f}")
            print(f"Best score so far: {best_score:.4f}")
            print(f"Total steps: {total_steps}")
            print("========================\n")

    # 训练结束，保存结果
    print("\n=== Training Completed ===")
    print(f"Total episodes: {N_Games}")
    print(f"Best score achieved: {best_score:.4f}")
    print(f"Average score: {np.mean(episode_rewards):.4f}")
    print(f"Average task accuracy: {np.mean(all_task_accuracy):.4f}")

    # 保存结果到文件
    with open(reward_dir, 'a') as file:
        file.write("# PPO Training Results\n")
        file.write(f"# Time: {current_time}\n")
        file.write(f"# Episodes: {N_Games}\n")
        file.write(f"# Best Score: {best_score}\n")
        file.write("\n".join(map(str, episode_rewards)) + '\n')

    with open(acc_dir, 'a') as file:
        file.write("# PPO Task Accuracy Results\n")
        file.write(f"# Time: {current_time}\n")
        file.write("\n".join(map(str, all_task_accuracy)) + '\n')

    with open(loss_dir, 'a') as file:
        file.write("# PPO Training Loss\n")
        file.write(f"# Time: {current_time}\n")
        file.write("\n".join(map(str, all_loss)) + '\n')
    wandb.finish()
