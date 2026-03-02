from model.CNN import CNNAnomalyDetector
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_data(file_path):
    data = torch.load(file_path)
    data = data.squeeze(2)
    data_reshape = data[:,0,:,:]
    return data_reshape.float()



def train_model(model, x, num_epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = x.permute(0, 2, 1).to(device)  # 转换形状为 [batch_size, chs, n_times]
    dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    return model



def calculate_threshold(model, x_train, batch_size = 32,q = 0.98):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.permute(0, 2, 1).to(device)

    model.eval()
    dataloader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        error_list = []
        for batch in dataloader:
            reconstructed = model(batch)
            error = torch.mean((batch - reconstructed) ** 2, dim=[0, 1, 2])
            error_list.append(error.item())

    errors = torch.tensor(error_list)
    threshold = torch.quantile(torch.tensor(errors), q)
    # threshold = torch.mean(errors) + 2 * torch.std(errors)
    print(f"Calculated Threshold: {threshold.item():.6f}")
    return threshold.item()


def detect_anomalies(model, data, threshold, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.permute(0, 2, 1).to(device)

    model.eval()
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        error_list = []
        for batch in dataloader:
            reconstructed = model(batch)
            error = torch.mean((batch - reconstructed) ** 2, dim=[0, 1, 2])
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
    pass

