from timeit import default_timer
import pandas as pd
import torch
import torch.nn as nn

df = pd.read_csv(r"./examples/housing.csv")
x = torch.tensor(df.iloc[:, :-1].to_numpy()).float()
y = torch.tensor(df.iloc[:, -1].to_numpy()).float().view(-1, 1)
mu = x.mean(0)
sigma = x.std(0)
x = (x - mu) / sigma

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__() 
        self.mlp = nn.Sequential(
            nn.Linear(13, 8),
            nn.ReLU(),
            nn.Linear(8, 1, bias=False),
        )
    
    def forward(self, x):
        return self.mlp(x)

mlp = MLP()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

start = default_timer()
for it in range(1000):
    loss = loss_fn(mlp(x), y)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Took {default_timer() - start} seconds.")