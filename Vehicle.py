import random


class Vec:
    def __init__(self, vec_id, location, velocity, direction):
        self.vec_id = vec_id
        self.location = location  # 当前坐标
        self.velocity = velocity  # 移动速度（以米/秒为单位）
        self.direction = direction  # 当前移动方向

    def move(self, grid_points, horizontal_lines, vertical_lines, time_slot, x_axis, y_axis, grid_size):
        """
        根据速度和方向移动车辆
        """
        x, y = self.location
        distance = self.velocity * time_slot  # 根据时隙计算移动距离

        # 判断车辆是否在水平线
        is_on_horizontal_line = any(
            abs(y - line_y) < 1e-6 and 0 <= x <= x_axis for line_y in range(0, y_axis + 1, int(y_axis / grid_size))
        )

        # 判断车辆是否在垂直线
        is_on_vertical_line = any(
            abs(x - line_x) < 1e-6 and 0 <= y <= y_axis for line_x in range(0, x_axis + 1, int(x_axis / grid_size))
        )

        # 根据当前位置限定方向
        if is_on_horizontal_line and not is_on_vertical_line:
            valid_directions = ["left", "right"]  # 水平线只允许左右移动
        elif is_on_vertical_line and not is_on_horizontal_line:
            valid_directions = ["up", "down"]  # 垂直线只允许上下移动
        else:
            valid_directions = []  # 如果不在线上，则没有有效方向

        # 如果方向不在有效方向内，重新选择一个有效方向
        if self.direction not in valid_directions:
            if valid_directions:
                self.direction = random.choice(valid_directions)

        # 根据当前方向计算下一位置
        if self.direction == "up":
            new_location = (x, y + distance)
        elif self.direction == "down":
            new_location = (x, y - distance)
        elif self.direction == "left":
            new_location = (x - distance, y)
        elif self.direction == "right":
            new_location = (x + distance, y)
        else:
            new_location = self.location  # 不动

        # 更新位置（确保新位置在网格范围内）
        if (
            self.direction in ["up", "down"] and 0 <= new_location[1] <= y_axis
            and any(abs(new_location[0] - line_x) < 1e-6 for line_x in range(0, x_axis + 1, int(x_axis / grid_size)))
        ):
            self.location = new_location
        elif (
            self.direction in ["left", "right"] and 0 <= new_location[0] <= x_axis
            and any(abs(new_location[1] - line_y) < 1e-6 for line_y in range(0, y_axis + 1, int(y_axis / grid_size)))
        ):
            self.location = new_location

        # 调试打印更新的位置
        #print(f"Vehicle {self.vec_id} moved to {self.location} with direction {self.direction}")
