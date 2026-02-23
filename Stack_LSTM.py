import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from prettytable import PrettyTable

class Stack_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Stack_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

def get_data(file_path):
    data = torch.load(file_path, map_location= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    data = data.squeeze(2)
    data_reshape = data[:, 0, :, :]
    return data_reshape.float()

# 训练函数
def train_model(model, data, learning_rate = 1e-4, batch_size = 32, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch[:, -1, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    return model

def calculate_threshold(model, x_train, batch_size = 32,q = 0.98):
    model.eval()
    dataloader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        error_list = []
        for batch in dataloader:
            outputs = model(batch)
            error = torch.mean((batch[:, -1, :] - outputs) ** 2, dim=[0, 1, 2])
            error_list.append(error.item())

    errors = torch.tensor(error_list)
    threshold = torch.quantile(torch.tensor(errors), q)
    # threshold = torch.mean(errors) + 2 * torch.std(errors)
    print(f"Calculated Threshold: {threshold.item():.6f}")
    return threshold.item()

def detect_anomalies(model, data, threshold, batch_size=32):
    model.eval()
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        error_list = []
        for batch in dataloader:
            outputs = model(batch)
            error = torch.mean((batch[:, -1, :] - outputs) ** 2, dim=[0, 1, 2])
            error_list.append(error.item())
    errors = torch.tensor(error_list)
    anomaly_labels = errors > threshold
    return anomaly_labels.cpu()

def print_table_result(abnormal_sum, normal_sum, abnormal_true, normal_true):
    acc = (normal_true + abnormal_true) / (abnormal_sum + normal_sum)
    precision_n = normal_true / (normal_true + abnormal_sum - abnormal_true)
    precision_a = abnormal_true / (abnormal_true + normal_sum - normal_true)
    pre_avg = (precision_a + precision_n) / 2
    recall_n = normal_true / normal_sum
    recall_a = abnormal_true / abnormal_sum
    recall_avg = (recall_a + recall_n) / 2
    F1 = 2 * (pre_avg * recall_avg) / (pre_avg + recall_avg)
    x = PrettyTable(["Acc", "Pre_Normal", "Pre_Abnormal", "Pre_avg", "Recall_Normal", "Recall_Abnormal", "Recall_avg", "F1"])
    row = [acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1]
    row = [round(i*100, 4) for i in row]
    x.add_row(row)
    print(x)



if __name__ == '__main__':
    folder_path = r"\PTB"
    x_train = get_data(os.path.join(folder_path, "x_train.pt"))
    x_test = get_data(os.path.join(folder_path, "x_test.pt"))
    abnormal = get_data(os.path.join(folder_path, "abnormal.pt"))
    batch_size = 32

    model = Stack_LSTM(input_size=x_train.shape[-1], hidden_size = 128, num_layers = 8, output_size = x_train.shape[-1])
    model = train_model(model, x_train, learning_rate = 1e-4, batch_size = batch_size, epochs=100)
    threshold = calculate_threshold(model, x_train, batch_size=batch_size, q = 0.98)
    normal_results = detect_anomalies(model, x_test, threshold, batch_size=batch_size)
    abnormal_results = detect_anomalies(model, abnormal, threshold, batch_size=batch_size)

    print(f"Normal Data Detected as Anomalies: {normal_results.sum().item()} / {len(normal_results)}")
    print(f"Anomalous Data Detected as Anomalies: {abnormal_results.sum().item()} / {len(abnormal_results)}")
    print_table_result(abnormal_sum = len(abnormal_results), normal_sum = len(normal_results),
                       abnormal_true = abnormal_results.sum().item(), normal_true = normal_results.sum().item())

