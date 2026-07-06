# %%
import torch
import torch.nn as nn
class TemporalAttention(nn.Module):

    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        H = self.hidden_size

        # Q, K, V 생성
        Q = self.query(x)  # [B, N, T, H]
        K = self.key(x)    # [B, N, T, H]
        V = self.value(x)  # [B, N, T, H]


        # T축 기준 attention 점수 계산
        # einsum: 'b n t h, b n s h -> b n t s'
        scores = torch.einsum('b t h, b s h -> b t s', Q, K) / (H ** 0.5)

        # T축에 softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N, T, T]

        # weighted sum
        out = torch.einsum('b t s, b s h -> b t h', attn_weights, V)  # [B, N, T, H]

        return out, attn_weights


# %%
import torch.nn as nn
class LSTM(nn.Module):
  def __init__(self, input_size = 30, hidden_size = 64, dropout = 0.0):
    super().__init__()
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.temp_attn = TemporalAttention(hidden_size= hidden_size)
    self.lstm1 = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, batch_first = True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):

    out, _ = self.lstm1(x)
    out = self.dropout(out)
    out, _ = self.lstm2(out)
    out, score = self.temp_attn(out)
    #out = out[:, -1, :]
    out = self.fc(out)
    return out

# %%
class FullModel(nn.Module):
    def __init__(self, input_tensor, hidden_size = 128):
        super().__init__()
        input_size = input_tensor.shape[-1]
        self.lstm = LSTM(input_size= input_size, hidden_size = hidden_size, dropout = 0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lstm(x)
        return out

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


