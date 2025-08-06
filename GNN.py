import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

# 检查是否有可用的GPU
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 权重初始化函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)


class GCNModel(T.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(GCNModel, self).__init__()
        # 增加网络深度和添加批归一化
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.batch_norm3 = nn.BatchNorm1d(output_dim)

        # 应用权重初始化
        self.apply(weights_init_)

        # 将模型移至GPU
        self.to(device)


        self.output_dim = output_dim


    def forward(self, x, edge_index):
        """
        前向传播，支持批量数据处理（逐图循环处理）。
        参数:
        - x: 节点特征张量，形状为 [batch_size, num_nodes, input_dim] 或 [num_nodes, input_dim]
        - edge_index: 边索引张量列表或单个边索引张量

        返回:
        - output: 节点表示，形状为 [batch_size, num_nodes, output_dim] 或 [num_nodes, output_dim]
        """
        # 确保输入张量在GPU上
        if isinstance(x, T.Tensor) and x.device != device:
            x = x.to(device)

        # 处理批量数据情况
        if len(x.size()) == 3:  # 如果输入是批量数据
            batch_size, num_nodes, feature_dim = x.size()
            outputs = []

            for i in range(batch_size):
                single_x = x[i]  # [num_nodes, input_dim]
                single_edge_index = edge_index[i]

                # 确保单个图的数据在GPU上
                if isinstance(single_x, T.Tensor) and single_x.device != device:
                    single_x = single_x.to(device)
                if isinstance(single_edge_index, T.Tensor) and single_edge_index.device != device:
                    single_edge_index = single_edge_index.to(device)

                # 第一层GCN + 残差连接
                h1 = self.conv1(single_x, single_edge_index)
                h1 = self.batch_norm1(h1)
                h1 = F.relu(h1)
                h1 = self.dropout1(h1)

                # 第二层GCN + 残差连接
                h2 = self.conv2(h1, single_edge_index)
                h2 = self.batch_norm2(h2)
                h2 = F.relu(h2)
                h2 = self.dropout2(h2)
                h2 = h2 + h1  # 残差连接

                # 第三层GCN
                h3 = self.conv3(h2, single_edge_index)
                h3 = self.batch_norm3(h3)
                # 最终残差连接 (需要投影以匹配维度)
                if single_x.size(1) == h3.size(1):
                    h3 = h3 + single_x  # 直接残差连接

                outputs.append(h3)  # 保存每个图的结果

            # 拼接所有图的输出
            output = T.stack(outputs, dim=0)  # [batch_size, num_nodes, output_dim]
            return output

        else:  # 如果输入是单图数据
            # 确保单图数据在GPU上
            if isinstance(edge_index, T.Tensor) and edge_index.device != device:
                edge_index = edge_index.to(device)

            # 第一层
            h1 = self.conv1(x, edge_index)
            h1 = self.batch_norm1(h1)
            h1 = F.relu(h1)
            h1 = self.dropout1(h1)

            # 第二层
            h2 = self.conv2(h1, edge_index)
            h2 = self.batch_norm2(h2)
            h2 = F.relu(h2)
            h2 = self.dropout2(h2)
            h2 = h2 + h1  # 残差连接

            # 第三层
            h3 = self.conv3(h2, edge_index)
            h3 = self.batch_norm3(h3)
            # 最终残差连接
            if x.size(1) == h3.size(1):
                h3 = h3 + x  # 直接残差连接

            return h3

    def save_checkpoint(self, path='./tmp/gcn.pth'):
        """
        保存 GCN 模型的参数
        """
        T.save(self.state_dict(), path)
        print(f"GCN model saved to {path}")

    def load_checkpoint(self, path='./tmp/gcn.pth'):
        """
        加载 GCN 模型的参数
        """
        try:
            # 加载到正确的设备上
            self.load_state_dict(T.load(path, map_location=device))
            print(f"GCN model loaded from {path}")
        except FileNotFoundError:
            print(f"Checkpoint not found at {path}. Model not loaded.")