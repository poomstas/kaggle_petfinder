# %%
# Fit a linear regression model to take a set of boolean metadata as inputs and predict pawpularity score. 
# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
TRAIN_CSV_PATH = '../data/train.csv'

df = pd.read_csv(TRAIN_CSV_PATH)
df

# %%
X_numpy = df.iloc[:, 1:13].to_numpy()
y_numpy = df.iloc[:, 13].to_numpy()

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # [1,2,3] -> [[1],[2],[3]]

n_samples, n_features = X.shape

print(n_features)

# %% Training Parameters
input_size = n_features
output_size = 1
learning_rate = 0.0003
n_epoch = 30000

# %% Define Model
class LinModel(nn.Module):
    def __init__(self, n_hidden=5):
        super(LinModel, self).__init__()
        self.fc1 = nn.Linear(12, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

model = LinModel()

# %% Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% Training Loop
for epoch in range(n_epoch):
    y_pred = model(X)
    loss = criterion(y, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("Epoch: {}\t Loss:{:.4f}".format(epoch+1, loss.item()))

# %% Calculate Validation Results
with torch.no_grad():
    predicted = model(X).detach()

# # %%
# plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.xlabel('X'); plt.ylabel('y')
# plt.legend(['Actual', 'Fitted'])
# plt.show()

# %%
plt.scatter(y, predicted); plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.show()
