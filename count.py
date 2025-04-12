import pickle

# 从 Pickle 文件中读取字典
with open("data.pkl", "rb") as file:
    loaded_data = pickle.load(file)

transition_keys = [
    "x", "edges", "network_state", "ready_mask", "distance_mask", "done_mask", "off_mask",
    "actions", "reward", "x_", "edges_", "new_network_state", "new_ready_mask",
    "new_distance_mask", "new_done_mask", "new_off_mask", "done", "acc", "delay"
]

reward = 1
punish = 1
import torch

base_acc = [0, 0, 0, 0, 0]
base_acc_tensor = torch.tensor(base_acc)

base_off_mask = [1, 1, 1, 1, 1]
base_off_mask_tensor = torch.tensor(base_off_mask)

for step, v in loaded_data.items():
    print(step, v)
    res = 0
    step_acc = v["acc"]
    delay = v["delay"]
    step_off_mask = v["off_mask"]
    step_acc_tensor = torch.tensor(step_acc)
    decision = v["decision"]
    step_off_mask_tensor = torch.tensor(step_off_mask)
    decision_tensor = decision.flatten()
    diff_acc = step_acc_tensor - base_acc_tensor
    diff_off_mask = base_off_mask_tensor - step_off_mask_tensor
    min_acc = step_acc_tensor.min()
    for i in range(len(step_acc)):
        if diff_acc[i] == 0 and diff_off_mask[i] == -1:
            # 差值=0 off_make由1->0停止更新
            print("停止更新")
            if delay > 0:
                if step_acc_tensor[i] > min_acc:
                    res = torch.nn.sigmoid(1 / (step_acc_tensor[i] - min_acc)) * reward
                elif step_acc_tensor[i] == min_acc:
                    res = 1 - punish
            elif delay == 0:
                if step_acc_tensor[i] > min_acc:
                    res = torch.nn.sigmoid(1 / (step_acc_tensor[i] - min_acc)) * punish * decision_tensor[i]
                elif step_acc_tensor[i] == min_acc:
                    res = 0
        if step_acc_tensor[i] > 0:
            if delay > 0:
                if step_acc_tensor[i] < min_acc:
                    res = torch.nn.sigmoid(1 / (step_acc_tensor[i] - min_acc)) * reward
                elif step_acc_tensor[i] > min_acc:
                    res = reward * decision_tensor[i]
            elif delay == 0:
                if step_acc_tensor[i] < min_acc:
                    res = reward * decision_tensor[i]
                elif step_acc_tensor[i] > min_acc:
                    res = punish * decision_tensor[i]
    print(res)

    # print(step, diff,delay)

    base_acc_tensor = step_acc_tensor
