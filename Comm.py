def calculate_power_for_all_subtasks(self, task, P_t=20, G_t=2, G_r=2, f=2.4e9):
    """
    为任务的每个子任务计算传输的接收功率
    :param task: 任务对象
    :param P_t: 发射功率 (dBm)
    :param G_t: 发射天线增益 (dBi)
    :param G_r: 接收天线增益 (dBi)
    :param f: 信号频率 (Hz)
    """
    for subtask in task.subtasks:
        # 获取发送方位置 (DT所在边缘服务器位置)
        sender_x, sender_y = subtask.dt_server.location

        # 获取接收方位置 (计算节点位置)
        receiver = subtask.compute_node
        if isinstance(receiver, eds.Edge):
            receiver_x, receiver_y = receiver.location
        elif isinstance(receiver, VE.Vec):
            receiver_x, receiver_y = receiver.location
        else:
            print(f"SubTask {subtask.subtask_id} has no assigned compute node.")
            continue

        # 计算接收功率
        received_power = self.calculate_received_power(P_t, G_t, G_r, sender_x, sender_y, receiver_x, receiver_y, f)
        print(f"SubTask {subtask.subtask_id} -> Compute Node {'Edge' if isinstance(receiver, eds.Edge) else 'Vehicle'} {receiver.edge_id if isinstance(receiver, eds.Edge) else receiver.vec_id}: Power = {received_power:.2f} dBm")
