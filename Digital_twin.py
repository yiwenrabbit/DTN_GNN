max_update_size = 20     #DT更新数据量上限
min_update_size = 10     #DT更新数据下限
import numpy as np

class DigitalT:
    def __init__(self, v_id):
        self.v_id = v_id
        self.update_size =  np.random.randint(min_update_size, max_update_size)
        self.current_update = 0

    def dt_update(self, size):
        #
        #更新当前DT的实时数据，并返回当前的精度上限
        #
        if self.current_update + size >= self.update_size:
            self.current_update = self.update_size
            return 1
        else:
            self.current_update += size
            return self.current_update/self.update_size

    def reset_update(self):
        #
        #当需要做新的子任务时，重置更新数据量
        #
        self.current_update = 0





