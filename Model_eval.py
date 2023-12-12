import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

test_df = pd.read_csv('test_data.csv')
test_df = test_df[test_df['wind_speed'] <= 10]
X_test = test_df[['Month', 'temp_celsius', 'humidity', 'wind_speed']].values
y_test = test_df['generation wind onshore'].values / 100.00

X_test_scaled = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

test_dataset = TensorDataset(X_test_scaled, y_test)

batch_size = 64
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

model_state = torch.load('model.pth')
model.load_state_dict(model_state)

model.eval()

columns = ['Month', 'temp_celsius', 'humidity', 'wind_speed', 'Truth', 'Prediction']
df = pd.DataFrame(columns=columns)

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_predictions = model(batch_X)
        concatenated_tensor = torch.cat((batch_X, batch_y, batch_predictions), dim=1)
        row = concatenated_tensor.numpy().reshape(-1) 
        temp_df = pd.DataFrame([row], columns=columns)
        df = pd.concat([df, temp_df], ignore_index=True)

print(df)

#시각화
predicted = [tensor.detach().numpy() for tensor in batch_predictions]
hourly_data = df.groupby('wind_speed')[['Truth', 'Prediction']].mean()
actuals = y_test.view(-1).tolist()
print(batch_predictions)


sns.barplot(x=hourly_data.index, y='Truth', data=hourly_data, color='green', label='Wind Generation')
sns.lineplot(x=hourly_data.index, y='Prediction', data=hourly_data, color='orange', marker="o", label='Wind Generation Prediction')
plt.title('Hourly Average Wind Generation and Load')
plt.xlabel('wind_speed')
plt.ylabel('Average Value')
plt.show()

#모델 평가

y_true = df['Truth'].astype(float).values
y_pred = df['Prediction'].astype(float).values

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred).round(2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse).round(2)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred).round(2)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
