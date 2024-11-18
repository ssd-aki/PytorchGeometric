import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# グラフデータセットの作成
graphs = []
for _ in range(100):  # ダミーデータとして100個のグラフを作成
    x = torch.randn((4, 2), dtype=torch.float)  # ノードの特徴量
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]], dtype=torch.long)  # エッジの接続関係
    graph = Data(x=x, edge_index=edge_index)
    graphs.append(graph)

dataset = graphs
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# モデルの初期化
in_channels = 2
hidden_channels = 4
out_channels = 2
gcn = GCN(in_channels, hidden_channels, out_channels)

# オプティマイザの設定
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)

# トレーニングループ
epochs = 1000
for epoch in range(epochs):
    gcn.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = gcn(batch.x, batch.edge_index)
        loss = F.mse_loss(out, batch.x)  # ダミーの損失関数として入力と出力のMSEを使用
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# グラフの定義
graph1_x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)  # ノードの特徴量
graph1_edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]], dtype=torch.long)  # エッジの接続関係
graph1 = Data(x=graph1_x, edge_index=graph1_edge_index)

graph2_x = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.float)  # ノードの特徴量
graph2_edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]], dtype=torch.long)  # エッジの接続関係
graph2 = Data(x=graph2_x, edge_index=graph2_edge_index)

# グラフの埋め込みを取得
graph1_embedding = gcn(graph1.x, graph1.edge_index)
graph2_embedding = gcn(graph2.x, graph2.edge_index)

# コサイン類似度を用いたノード間のマッチング
similarity_matrix = F.cosine_similarity(graph1_embedding.unsqueeze(1), graph2_embedding.unsqueeze(0), dim=-1)

# 最大の類似度を持つノードをマッチング
matching = torch.argmax(similarity_matrix, dim=1)

# 結果の出力
for i, j in enumerate(matching):
    print(f"Graph1 Node {i} is matched with Graph2 Node {j.item()}")