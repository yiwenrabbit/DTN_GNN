from environment import Env
import environment as ENVN
from agent import Agent
from buffer import ReplayBuffer
from GNN import GCNModel
import matplotlib
import numpy as np
import torch as t

def obs_list_to_state_vector(observation):  # 将观察聚合为state
    obs_clone = observation.clone()
    state = np.array(obs_clone)
    #state = np.concatenate(state, axis=0)
    return state

max_delay = ENVN.max_delay





# 测试任务类
if __name__ == "__main__":
    # 参数设置
    state_dim = 10
    subtask_num = ENVN.subtasks_dag      #一个任务有多少子任务

    max_size = 1000
    batch_size = 64
    edge_num = 16  # 假设有16个边缘节点
    vehicle_num = 20  # 假设有20辆车

    ###GCN参数设定#####################
    Gcn_input_dim = 3 + 2 * edge_num
    Gcn_hidden_dim = 32
    Gcn_output_dim = 16

    GCN_model = GCNModel(input_dim=Gcn_input_dim, hidden_dim=Gcn_hidden_dim, output_dim=Gcn_output_dim)

    ###DRL参数设定#####################
    state_dim = subtask_num * Gcn_output_dim + 1 #GCN的输出特征+任务总时延
    n_actions = int(sum(np.ones(subtask_num)*(edge_num+1)))   #每个任务输出DT放置的概率分布，以及一个数据更新的百分比

    ###加载环境
    env = Env.load_env()
    _, edges = env.get_gcn_inputs()
    edge_dim = edges.shape[1]
    distance_mask_dim =  edge_num
    ready_mask_dim = subtask_num

    # 初始化 Agent 和 ReplayBuffer
    DT_place_agent = Agent(actor_dims=state_dim, critic_dims=state_dim, gcn_input_dim=Gcn_input_dim, gcn_hidden_dim=Gcn_hidden_dim, gcn_output_dim=Gcn_output_dim, n_actions=n_actions, agent_idx=0, n_subtasks= subtask_num, n_edges=edge_num,
                  chkpt_dir='./tmp/', alpha=0.0001, beta=0.0001, fc1=256, fc2=128, fc3=64)
    buffer = ReplayBuffer(max_size=max_size, x_dim=Gcn_input_dim, node_count=subtask_num ,edge_dim=edge_dim, action_dim=n_actions, batch_size=batch_size, ready_mask_dim=ready_mask_dim,distance_mask_dim=distance_mask_dim)

    ###游戏开始####
    N_Games = 2       #进行多少次
    MAX_STEPS = 400      #最多的时隙数
    total_steps = 0
    best_score = 0

    for i in range(N_Games):  #obs是当前的观察，obs_是做完动作后的观察
        if i==0:
            pass
        else:
            env = env.reset_env()

        score = 0
        done = [False]
        episode_step = 1
        done = [False]
        while not any(done):

            #获取节点和边
            x, edges = env.get_gcn_inputs()

            # 假设任务剩余完成时间
            task_delay = env.tasks[0].task_delay / max_delay
            remaining_time = t.tensor([task_delay])  # 剩余完成时间 (标量值)
            ready_mask, distance_mask = env.generate_masks()

            actions = DT_place_agent.choose_action(x, edges, remaining_time, total_steps, ready_mask, distance_mask)

            reward, x_, edges_, new_remaining_time, new_ready_mask, new_distance_mask = env.step(actions)

            if episode_step >= MAX_STEPS or env.tasks[0].task_delay==0 or env.tasks[0].is_completed:
                done = [True]
                ave_re = (score + reward) / MAX_STEPS

            buffer.store_transition(x, edges, remaining_time, ready_mask, distance_mask,
                                    actions, reward, x_, edges_, new_remaining_time, new_ready_mask, new_distance_mask, done)


            DT_place_agent.learn(buffer, total_steps)

            score += reward
            step_reward = reward
            print(f"########Current episode: {episode_step}########")
            print(f"Reward: {step_reward}, and informations:")

            #sheet.write(i, episode_step-1, step_reward)

            episode_step += 1
            total_steps += 1

            # if c_tmp:
            #     for c in range(len(c_tmp)):
            #         loss_jyz.append(c_tmp[c])

            if episode_step == MAX_STEPS + 1:
                print(i)


        if score > best_score:
            DT_place_agent.save_models()
            best_score = score
