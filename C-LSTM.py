import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from prettytable import PrettyTable


class C_LSTM(nn.Module):
    def __init__(self, input_channels, n_times, lstm_hidden_size, lstm_layers, num_classes):
        super(C_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.lstm = nn.LSTM(input_size=(input_channels // 4) * (n_times // 4), hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, chs = x.shape  # (batch_size, n_times, chs)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (batch_size, 32, new_time, new_chs)
        x = x.view(batch_size, 32, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def get_data(file_path, cuda_id = 0):
    data = torch.load(file_path, map_location= torch.device(f"cuda:{cuda_id}"))
    data = data.squeeze(2)
    data_reshape = data[:, 0, :, :]
    return data_reshape.float()

def print_table_result(abnormal_sum, normal_sum, abnormal_true, normal_true):
    acc = (normal_true + abnormal_true) / (abnormal_sum + normal_sum)
    precision_n = normal_true / (normal_true + abnormal_sum - abnormal_true)
    precision_a = abnormal_true / (abnormal_true + normal_sum - normal_true)
    pre_avg = (precision_a + precision_n) / 2
    recall_n = normal_true / normal_sum
    recall_a = abnormal_true / abnormal_sum
    recall_avg = (recall_a + recall_n) / 2
    F1 = 2 * (pre_avg * recall_avg) / (pre_avg + recall_avg + 1e-10)
    x = PrettyTable(["Acc", "Pre_Normal", "Pre_Abnormal", "Pre_avg", "Recall_Normal", "Recall_Abnormal", "Recall_avg", "F1"])
    row = [acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1]
    row = [round(i*100, 4) for i in row]
    x.add_row(row)
    print(x)

def train(model, data, cuda_id = 0, learning_rate = 1e-4, batch_size = 32, epochs=100):
    device = torch.device(f"cuda:{cuda_id}")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            labels = torch.zeros(batch.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.10f}")
    return model


def test(model, x_test, abnormal, cuda_id = 0):
    device = torch.device(f"cuda:{cuda_id}")
    model.to(device)
    model.eval()

    with torch.no_grad():
        normal_sum = torch.argmax(F.softmax(model(x_test), dim=1), dim=1)
        abnormal_sum = torch.argmax(F.softmax(model(abnormal), dim=1), dim=1)


    normal_true = (normal_sum == 0).sum().item()
    abnormal_true = (abnormal_sum == 1).sum().item()

    print(f"Normal Data Detected as Anomalies: {normal_true} / {len(normal_sum)}")
    print(f"Anomalous Data Detected as Anomalies: {abnormal_true} / {len(abnormal_sum)}")


    return len(abnormal_sum), len(normal_sum), abnormal_true, normal_true


if __name__ == "__main__":
    pass


