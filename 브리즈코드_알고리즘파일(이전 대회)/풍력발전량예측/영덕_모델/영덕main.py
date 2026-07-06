# %%
from . import load_preprocess
target = load_preprocess.target
trains = load_preprocess.trains
tests = load_preprocess.tests

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from . import scale_tensor
X_trains, X_valids, y_trains, y_valids, X_tests, X_alls = scale_tensor.train_valid_split(trains, tests)
X_train, X_valid, y_train, y_valid, X_test, X_all = scale_tensor.to_tensor(X_trains, X_valids, y_trains, y_valids, X_tests, X_alls)

# %%
X_train = X_train.to(device)
X_valid = X_valid.to(device)
y_train = y_train.to(device)
y_valid = y_valid.to(device)
X_test = X_test.to(device)
X_all = X_all.to(device)

# %%
from . import 영덕model
import torch.nn as nn

latitudes = 영덕model.latitudes
longitudes = 영덕model.longitudes

model = 영덕model.FullModel(latitudes, longitudes, X_train)
model = model.to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0015)  # 모델의 loss가 한값에 고정되어 나오면 lr을 조금씩 수정해주세요
early_stopping = 영덕model.EarlyStopping(patience=30, verbose=True, delta=0.0001)

영덕model.train_loop(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=500,
    early_stopping=early_stopping,
    input_tensor=X_train,
    valid_tensor=X_valid,
    tg=y_train,
    vt=y_valid
)


# %%
import pandas as pd
import matplotlib.pyplot as plt

#Validation용 2023년 예측 및 시각화

model.eval()
with torch.no_grad():
    predictions  = model(X_valid)
    predictions_np = predictions.cpu().numpy() 

output = predictions_np
output = output.reshape(-1, 1)

output_inverse = scale_tensor.target_scaler.inverse_transform(output)
output_inverse = output_inverse.reshape(-1, 1)

pred = pd.DataFrame(output_inverse, columns = ["pred"])


target2023 = target.loc["2024-12-01":]
pred.index = trains["wtg_1"].loc["2024-12-01 01:00:00":].index
pred_loc = pred.loc["2025-01-01 01:00:00":"2025-03-31 23:00:00"]
target2023 = target2023.loc["2025-01-01 01:00:00":"2025-03-31 23:00:00"]

plt.figure(figsize = (12,4))
plt.plot(pred_loc.index, pred_loc, label = "pred")
plt.plot(target2023.index, target2023, label = "target", alpha = 0.4)
plt.legend()
plt.tight_layout()
plt.show()

# %%
pred_loc = pred_loc.fillna(0)
target2023 = target2023.fillna(0)
pred_loc.index = target2023.index

# %%
#대회 전용 오차
#발전용량으로 정규화한 mae

#실제 센서데이터 사용시 NMAE 3~4 (사용 불가능. 2024년의 센서데이터가 없음)
#예측 센서데이터 사용시 NMAE 12~14(사용 가능)

def nmae(target, pred_target):
  mae = abs((target - pred_target))/42000
  nmae1 = mae.mean() * 100
  return nmae1

nmae1 = nmae(target2023["energy_kwh"], pred_loc["pred"])
print(nmae1)

# %%
import pandas as pd
import matplotlib.pyplot as plt

#Validation용 2023년 예측 및 시각화

model.eval()
with torch.no_grad():
    predictions  = model(X_test)
    predictions_np = predictions.cpu().numpy() 

output = predictions_np
output = output.reshape(-1, 1)

output_inverse = scale_tensor.target_scaler.inverse_transform(output)
output_inverse = output_inverse.reshape(-1, 1)

pred = pd.DataFrame(output_inverse, columns = ["pred"])


pred.index = trains["wtg_1"].index

plt.figure(figsize = (12,4))
plt.plot(pred.index, pred, label = "pred")
plt.legend()
plt.tight_layout()
plt.show()

# %%
pred.to_csv("./영덕_모델/영덕_예측/영덕_pred.csv")


