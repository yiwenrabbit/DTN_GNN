import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)



    def forward(self, x, edge_index):
        """
        前向传播，支持批量数据处理（逐图循环处理）。
        参数:
        - x: 节点特征张量，形状为 [batch_size, num_nodes, input_dim]
        - edge_index: 边索引张量列表，长度为 batch_size，每个元素是 [2, num_edges]

        返回:
        - output: 合并后的节点特征，形状为 [batch_size, num_nodes, output_dim]
        """
        if len(x.size()) == 3:  # 如果输入是批量数据
            batch_size, num_nodes, _ = x.size()
            outputs = []

            for i in range(batch_size):
                single_x = x[i]  # 获取单个图的节点特征，形状为 [num_nodes, input_dim]
                single_edge_index = edge_index[i]  # 获取单个图的边索引，形状为 [2, num_edges]

                # 单图的图卷积处理
                h = self.conv1(single_x, single_edge_index)
                h = torch.relu(h)
                h = self.conv2(h, single_edge_index)
                outputs.append(h)  # 保存每个图的结果

            # 拼接所有图的输出
            output = torch.stack(outputs, dim=0)  # [batch_size, num_nodes, output_dim]
            return output
        else:  # 如果输入是单图数据
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            return h

    def save_checkpoint(self, path='./tmp/gcn.pth'):
        """
        保存 GCN 模型的参数
        """
        torch.save(self.state_dict(), path)
        print(f"GCN model saved to {path}")

    def load_checkpoint(self, path='./tmp/gcn.pth'):
        """
        加载 GCN 模型的参数
        """
        try:
            self.load_state_dict(torch.load(path))
            print(f"GCN model loaded from {path}")
        except FileNotFoundError:
            print(f"Checkpoint not found at {path}. Model not loaded.")




