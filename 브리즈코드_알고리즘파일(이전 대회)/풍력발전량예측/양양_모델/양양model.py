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

latitudes = [37.943167, 37.941000, 37.939139, 37.937361, 37.934111, 37.931833, 37.929806, 37.927333, 37.926167, 37.924028]

longitudes = [128.692028, 128.693250, 128.692917, 128.694306, 128.694500, 128.695306, 128.694306, 128.693750, 128.695694, 128.695889]


def build_geo_adjacency(latitudes, longitudes, threshold_km = 1.0):
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

adjacency = build_geo_adjacency(latitudes, longitudes)
print(f"Adjacency Matrix \n{adjacency}")


# %%
"""
import torch.nn as nn

class LSTMgcn(nn.Module):
  def __init__(self, hidden_size = 64, dropout = 0.0):
    super().__init__()
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm1 = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, batch_first = True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    N = x.shape[1]
    datas = []
    for i in range(N):
      data = x[:, i, :, :]
      out, _ = self.lstm1(data)
      out = self.fc(out)
      datas.append(out.unsqueeze(1))
    out = torch.cat(datas, dim = 1)
    out = torch.mean(out, dim = 1) # B T 1
    return out
"""

# %%
import torch.nn as nn

class LSTMgcn(nn.Module):
  def __init__(self, input_size = 32, hidden_size = 64, dropout = 0.0):
    super().__init__()
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm1 = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    N = x.shape[1]
    datas = []
    for i in range(N):
      data = x[:, i, :, :]
      out, _ = self.lstm1(data)
      out = self.fc(out)
      datas.append(out.unsqueeze(1))
    out = torch.cat(datas, dim = 1)
    out = torch.mean(out, dim = 1) # B T 1
    return out

# %%
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
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

    def forward(self, x, adj, weights):
        # x: [B, N, T, H]
        # adj: [B, N, N]

        B, N, T, H = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, T, N, H]


        outputs = []
        #시간축마다 convolution 진행
        for t in range(T):
            xt = x[:, t, :, :]  # [B,T(1~24), N, H]
            weight = weights[:, t, :, :]  # [B, T(1~24), N, N]

            support = self.linear(xt)
            weighted_adj = adj * weight
            out = torch.bmm(weighted_adj, support)  # [B, N, out_features]
            outputs.append(out.unsqueeze(1))  # [B,1,N,out_features]

        out = torch.cat(outputs, dim=1)  # [B, T, N, out_features]
        out = out.permute(0, 2, 1, 3)  # [B, N, T, out_features]

        return out

# %%
class FullModel(nn.Module):
    def __init__(self, latitudes, longitudes, input_tensor, hidden_size = 64, gcn_hidden= 64):
        super().__init__()
        input_size = input_tensor.shape[-1]
        self.lstm = LSTMgcn(input_size = input_size, hidden_size = hidden_size, dropout = 0.2)
        #self.temporal_attn = TemporalAttention(input_size = input_size, hidden_size= hidden_size)
        #self.spatial_attn = SpatialAttention(input_size = input_size, hidden_size = hidden_size)
        #self.gcn = GCNLayer(in_features=hidden_size, out_features= hidden_size)

        lat_rad = torch.tensor(latitudes).float() * torch.pi / 180
        lon_rad = torch.tensor(longitudes).float() * torch.pi / 180
        self.register_buffer('latitudes', lat_rad)
        self.register_buffer('longitudes', lon_rad)

        #A_geo = build_geo_adjacency(self.latitudes, self.longitudes).to(torch.float32)
        #self.register_buffer('A_geo', A_geo)
        self.relu = nn.ReLU()





    def forward(self, x):
        B = x.size(0)
        F_in = x.size(3)
        N = x.size(1)
        T = x.size(2)


        #temp_out, temp_weights = self.temporal_attn(x)  # [B, N, T, H], [B, N, T, 1]
        #spatial_out, spatial_weights = self.spatial_attn(x)  # [B, N, T, H], [B, N, T, N]

        #A_geo_expanded = self.A_geo.unsqueeze(0).expand(B, -1, -1)

        #gcn_out = self.gcn(temp_out, A_geo_expanded, spatial_weights)  # [B, N, T, 64]

        lstm_out = self.lstm(x)

        out_hard = self.relu(lstm_out)
        return out_hard

# %%
import torch
import copy

class EarlyStopping:
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
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        predictions = model(input_tensor)
        predictions = predictions.squeeze(1).squeeze(-1) # B T

        loss = criterion(predictions, tg.squeeze(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


        model.eval()
        val_loss = 0
        with torch.no_grad():
            predictions_val = model(valid_tensor)
            predictions_val = predictions_val.squeeze(1).squeeze(-1)
            loss_hard_val = criterion(predictions_val, vt.squeeze(-1))
            loss = loss_hard_val
            val_loss += loss.item()


        early_stopping(val_loss, model, current_epoch = epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.4f}")
    print(f"Best performance at epoch {early_stopping.best_epoch} with validation loss {-early_stopping.best_score:.6f}")
    model.load_state_dict(early_stopping.best_model_state)


