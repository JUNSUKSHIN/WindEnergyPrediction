import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
train_df = train_df[train_df['wind_speed'] <= 10]
test_df = test_df[test_df['wind_speed'] <= 10]

X_train = train_df[['Month', 'temp_celsius', 'humidity', 'wind_speed']].values
y_train = train_df['generation wind onshore'].values / 100.00
X_test = test_df[['Month', 'temp_celsius', 'humidity', 'wind_speed']].values
y_test = test_df['generation wind onshore'].values / 100.00

X_train_scaled = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_scaled = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_scaled, y_train)
test_dataset = TensorDataset(X_test_scaled, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(4,8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = RegressionModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 10
early_stopping_limit = 10
best_val_loss = float("inf")
no_improvement_count = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader: 
        
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            val_predictions = model(batch_X)
            val_loss = criterion(val_predictions, batch_y)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    current_lr = optimizer.param_groups[0]['lr']

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
    else:
        no_improvement_count += 1
            
torch.save(best_model_state, 'best_model.pth')

