#class Edge():
   # def __init__(self, location):
        #self.location = location
        #self.set_dts = []               #set of digital twins
class EdgeServer:
    def __init__(self, server_id, location, total_computation_resources):
        """
        初始化边缘服务器
        :param server_id: 边缘服务器的唯一标识
        :param location: 边缘服务器的位置 (x, y)
        :param total_computation_resources: 边缘服务器的总计算资源 (CPU cycles)
        """
        self.server_id = server_id
        self.location = location  # 位置 (x, y)
        self.total_computation_resources = total_computation_resources
        self.available_resources = total_computation_resources  # 当前可用资源
        self.dt_storage = {}  # 存放车辆 DT 的字典 {dt_id: DT object}

    def store_dt(self, dt):
        """
        将数字孪生(DT)存储到边缘服务器
        :param dt: 数字孪生对象 (需要有属性 dt_id)
        """
        if dt.dt_id in self.dt_storage:
            print(f"DT {dt.dt_id} 已经存储在边缘服务器 {self.server_id} 上")
        else:
            self.dt_storage[dt.dt_id] = dt
            print(f"DT {dt.dt_id} 成功存储到边缘服务器 {self.server_id} 上")

    def remove_dt(self, dt_id):
        """
        从边缘服务器移除数字孪生(DT)
        :param dt_id: 需要移除的DT的ID
        """
        if dt_id in self.dt_storage:
            del self.dt_storage[dt_id]
            print(f"DT {dt_id} 已从边缘服务器 {self.server_id} 移除")
        else:
            print(f"DT {dt_id} 不存在于边缘服务器 {self.server_id} 上")

    def allocate_resources(self, resources_needed):
        """
        为任务分配计算资源
        :param resources_needed: 任务所需的计算资源
        :return: 是否分配成功 (True/False)
        """
        if resources_needed <= self.available_resources:
            self.available_resources -= resources_needed
            print(f"已分配 {resources_needed} 计算资源，服务器 {self.server_id} 剩余资源：{self.available_resources}")
            return True
        else:
            print(f"服务器 {self.server_id} 资源不足，无法分配 {resources_needed} 计算资源")
            return False

    def release_resources(self, resources_released):
        """
        释放计算资源
        :param resources_released: 释放的计算资源
        """
        self.available_resources += resources_released
        if self.available_resources > self.total_computation_resources:
            self.available_resources = self.total_computation_resources
        print(f"已释放 {resources_released} 计算资源，服务器 {self.server_id} 当前可用资源：{self.available_resources}")

    def get_dt_list(self):
        """
        获取当前存储的所有DT的ID列表
        :return: DT ID列表
        """
        return list(self.dt_storage.keys())

    def __repr__(self):
        """
        边缘服务器的字符串表示
        """
        return (f"EdgeServer(server_id={self.server_id}, location={self.location}, "
                f"total_resources={self.total_computation_resources}, available_resources={self.available_resources}, "
                f"stored_DT_count={len(self.dt_storage)})")
