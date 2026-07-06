# %%
import torch
def haversine_distance(lat1, lon1, lat2, lon2): #Haversine 거리를 이용해 인접행렬 만들기 위한 정보 생성
  R = 6371.0
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = torch.sin(dlat / 2) **2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) **2
  c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
  distance = R * c
  return distance

# %%
def build_geo_adjacency(threshold_km = 1.0):
  latitudes = [35.724089, 35.722233, 35.721336, 35.719208, 35.716156, 35.712278, 35.709742, 35.707047, 35.701786]
  longitudes = [129.374592, 129.3724, 129.37015, 129.368869, 129.367767, 129.367161, 129.367522, 129.366964, 129.368639]
  N = len(latitudes)
  lat = torch.tensor(latitudes).float() * torch.pi / 180
  lon = torch.tensor(longitudes).float() * torch.pi / 180

  lat1 = lat.unsqueeze(0).repeat(N, 1)
  lon1 = lon.unsqueeze(0).repeat(N, 1)
  lat2 = lat.unsqueeze(1).repeat(1, N)
  lon2 = lon.unsqueeze(1).repeat(1, N)

  dist_matrix = haversine_distance(lat1, lon1, lat2, lon2)
  adjacency = (dist_matrix <= threshold_km).float()
  adjacency.fill_diagonal_(1) #Self loop 추가
  return adjacency

# %%
adjacency = build_geo_adjacency(threshold_km= 1.0)
print(f"Adjacency Matrix \n{adjacency}")

# %%
import torch.nn as nn
class LSTMgcn(nn.Module):
  def __init__(self, hidden_size = 64, dropout = 0.0):
    super().__init__()
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm1 = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = hidden_size, hidden_size = 64, batch_first = True)
    self.lstm3 = nn.LSTM(input_size = 64, hidden_size = 32, batch_first = True)
    self.fc = nn.Linear(32, hidden_size)

  def forward(self, x):
    N = x.shape[1]
    datas = []
    for i in range(N):
      data = x[:, i, :, :]
      out, _ = self.lstm1(data)
      out = self.dropout(out)
      out, _ = self.lstm2(out)
      out = self.dropout(out)
      out, _ = self.lstm3(out)
      out = self.fc(out)
      datas.append(out.unsqueeze(1))
    out = torch.cat(datas, dim = 1)
    out = torch.mean(out, dim = 1).unsqueeze(1) # B 1 T H
    return out

# %%
class TemporalAttention(nn.Module):
    import torch
    import torch.nn as nn
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        B, N, T, F = x.shape

        H = self.hidden_size

        # Q, K, V 생성
        Q = self.query(x)  # [B, N, T, H]
        K = self.key(x)    # [B, N, T, H]
        V = self.value(x)  # [B, N, T, H]


        # T축 기준 attention 점수 계산
        # einsum: 'b n t h, b n s h -> b n t s'
        scores = torch.einsum('b n t h, b n s h -> b n t s', Q, K) / (H ** 0.5)

        # T축에 softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N, T, T]

        # weighted sum
        out = torch.einsum('b n t s, b n s h -> b n t h', attn_weights, V)  # [B, N, T, H]

        return out, attn_weights


# %%
class SpatialAttention(nn.Module):
    def __init__(self, input_size = 32, hidden_size= 64):
        super().__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: [B, N, T, H]
        H = self.hidden_size
        B, N, T, F = x.shape
        q = self.query(x)  # [B, N, T, H]
        k = self.key(x)    # [B, N, T, H]
        v = self.value(x)

        # 내적을 위한 차원축 변형
        q_reshaped = q.permute(0, 2, 1, 3) # [B, T, N, H]
        k_reshaped = k.permute(0, 2, 1, 3) # [B, T, N, H]
        v_reshaped = v.permute(0, 2, 1, 3) # [B, T, N, H]

        # Softmax(QK^T/d^-1/2)
        scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (H ** 0.5) # [B, T, N, N]
        attn_weights = torch.softmax(scores, dim=3) # [B, T, N, N]

        out = torch.matmul(attn_weights, v_reshaped) # [B, T, N, H]

        out = out.permute(0, 2, 1, 3) # [B, N, T, H]

        return out, attn_weights

# %%
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # x: [B, N, T, H]
        # adj: [B, N, N]

        B, N, T, H = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, T, N, H]
        out = torch.einsum('b t n n, b t n h -> b t n h', adj, x)

        out = out.permute(0, 2, 1, 3)  # [B, N, T, out_features]
        out = self.linear(out)
        out = self.leaky_relu(out)

        return out

# %%
class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, N, T, 1]
        out = self.fc(x)  # [B, N, T, 1]
        return out # [B, N, T, 1]

# %%
class FullModel(nn.Module):
    def __init__(self, input_tensor, hidden_size = 128, gcn_hidden= 128):
        super().__init__()
        input_size = input_tensor.shape[-1]
        self.lstm = LSTMgcn(hidden_size = hidden_size, dropout = 0.3)
        self.temporal_attn = TemporalAttention(input_size = hidden_size, hidden_size= hidden_size)
        self.spatial_attn = SpatialAttention(input_size = input_size, hidden_size = hidden_size)
        self.gcn = GCNLayer(in_features=input_size, out_features= hidden_size)
        self.output_layer = OutputLayer(input_dim=gcn_hidden)

        A_geo = build_geo_adjacency(threshold_km= 1.0).to(torch.float32)
        self.register_buffer('A_geo', A_geo)
        self.relu = nn.ReLU()



    def forward(self, x):
        B = x.size(0)
        F_in = x.size(3)
        N = x.size(1)
        T = x.size(2)


        spatial_out, spatial_weights = self.spatial_attn(x)  # [B, N, T, H], [B, T, N, N]

        A_geo = self.A_geo

        A_geo_attn = spatial_weights * A_geo
        A_geo_attn = torch.where(A_geo_attn == 0, float('-inf'), A_geo_attn)
        A_geo_attn = torch.softmax(A_geo_attn, dim = -1)


        gcn_out = self.gcn(x, A_geo_attn)  # [B, N, T, 64]

        lstm_out = self.lstm(gcn_out)

        temp_out, temp_weights = self.temporal_attn(lstm_out)  # [B, N, T, H], [B, N, T, T]

        output = self.output_layer(temp_out)  # [B, N, T, 1]


        out_hard = self.relu(output)
        return out_hard, A_geo_attn

# %%
class EarlyStopping:
    import torch
    def __init__(self, patience=5, verbose=False, delta=0, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = None  # 🔹 추가

    def __call__(self, val_metric, model, current_epoch):
        import copy
        # mode에 따라 score 계산
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = current_epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = current_epoch  # 🔹 가장 좋은 epoch 업데이트
            self.counter = 0


# %%
def train_loop(
        input_tensor,
        valid_tensor,
        tg,
        vt,
        model,
        criterion,
        optimizer,
        num_epochs,
        early_stopping
        ):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        predictions, adj = model(input_tensor)
        predictions = predictions.squeeze(1).squeeze(-1) # B T

        loss = criterion(predictions, tg.squeeze(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


        model.eval()
        val_loss = 0
        with torch.no_grad():
            predictions_val, adj_val = model(valid_tensor)
            predictions_val = predictions_val.squeeze(1).squeeze(-1)
            loss_hard_val = criterion(predictions_val, vt.squeeze(-1))
            loss = loss_hard_val
            val_loss += loss.item()


        early_stopping(val_loss, model, current_epoch = epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            print(torch.mean(adj_val, dim = (0, 1)))
            break

        print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.4f}")
    print(f"Best performance at epoch {early_stopping.best_epoch} with validation loss {-early_stopping.best_score:.6f}")
    model.load_state_dict(early_stopping.best_model_state)


