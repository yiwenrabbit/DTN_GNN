max_update_size = 200     #DT更新数据量上限
min_update_size = 100     #DT更新数据下限
import numpy as np

class DigitalT:
    def __init__(self, v_id):
        self.v_id = v_id
        self.update_size =  np.random.randint(min_update_size, max_update_size)
        self.current_update = 0
        self.update_done = False

    def dt_update(self, size):
        #
        #更新当前DT的实时数据，并返回当前的精度上限

        if self.current_update + size >= self.update_size:
            self.current_update = self.update_size
            print(f"DT of {self.v_id} finished updating.")
            return 1
        else:
            self.current_update += size
            accuracy = self.current_update/self.update_size
            print(f"DT of {self.v_id} is updating, Accuracy: {accuracy:.3f}")
            return self.current_update/self.update_size

    def if_dy_update(self, accuracy):
        if self.current_update / self.update_size >= accuracy:
            print("DT update satisfies requirement")
            self.update_done = True


    def reset_update(self):
        #
        #当需要做新的子任务时，重置更新数据量
        #
        self.current_update = 0





