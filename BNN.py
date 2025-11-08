# --------------------------------------------------------------
# BNN via MC Dropout
# --------------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # <-- ADD THIS LINE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
X = torch.linspace(-1, 1, 100).unsqueeze(1)
y = X**3 + 0.1 * torch.randn_like(X)

# --- Neural Net with Dropout (BNN) ---
class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(p=0.3)  # <-- BNN!

    def forward(self, x, train=True):
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if train else x  # Only during training & MC
        x = self.fc2(x)
        return x

# --- Train ---
model = BNN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Training...")
for step in range(2001):
    opt.zero_grad()
    pred = model(X, train=True)
    loss = F.mse_loss(pred, y)
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step:4d} loss {loss.item():.2f}")

# --- MC Dropout Prediction (100 samples) ---
model.eval()
model.train()  # Keep dropout ON for MC sampling!
with torch.no_grad():
    preds = [model(torch.linspace(-1.5, 1.5, 200).unsqueeze(1), train=True) for _ in range(1000)]
    preds = torch.stack(preds).squeeze(-1).numpy()  # (100, 200)

mean = preds.mean(0)
lower = np.percentile(preds, 5, axis=0)
upper = np.percentile(preds, 95, axis=0)

# --- Plot ---
X_test = np.linspace(-1.5, 1.5, 200)
plt.figure(figsize=(8, 5))
plt.scatter(X, y, c='black', s=20, label='data')
plt.plot(X_test, mean, 'red', lw=2, label='BNN mean')
plt.fill_between(X_test, lower, upper, color='red', alpha=0.3, label='90% PI')
plt.axvline(-1, color='gray', ls='--'); plt.axvline(1, color='gray', ls='--')
plt.xlabel('x'); plt.ylabel('y')
plt.title('BNN via MC Dropout')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()