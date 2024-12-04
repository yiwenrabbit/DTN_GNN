import environment as ENVN
import matplotlib

# 测试任务类
if __name__ == "__main__":
    edge_num = 16  # 假设有16个边缘节点
    vehicle_num = 20  # 假设有20辆车

    # 加载已有车辆位置
    load_from_file = True  # 设置为 False 以重新生成车辆并保存
    #matplotlib.use("TkAgg")
    #env = Env(edge_num=edge_num, vehicle_num=vehicle_num)
    env = ENVN.Env(edge_num=edge_num, vehicle_num=vehicle_num, load_from_file=load_from_file)
    env.display_vehicles()
    env.plot_vehicle_positions()

    for task in env.tasks:
        task.plot_dependency_graph()
