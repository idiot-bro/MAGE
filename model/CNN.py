import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from prettytable import PrettyTable
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

class CNNAnomalyDetector(nn.Module):
    def __init__(self, chs):
        super(CNNAnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=chs, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, chs, kernel_size=5, stride=2, padding=2, output_padding=1),
        )


    def forward(self, x):
        # print("Input shape:", x.shape)
        encoded = self.encoder(x)
        # print("Encoded shape:", encoded.shape)
        decoded = self.decoder(encoded)
        # print("Decoded shape:", decoded.shape)
        return decoded
if __name__ == '__main__':
    def get_data(file_path):
        data = torch.load(file_path)
        data = data.squeeze(2)
        data_reshape = data[:, 0, :, :]
        return data_reshape.float()


    def train_model(model, x, num_epochs=100, batch_size=32, learning_rate=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        x = x.permute(0, 2, 1).to(device)
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


    def calculate_threshold(model, x_train, batch_size=32, q=0.98):
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
        x = PrettyTable(
            ["Acc", "Pre_Normal", "Pre_Abnormal", "Pre_avg", "Recall_Normal", "Recall_Abnormal", "Recall_avg", "F1"])
        row = [acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1]
        row = [round(i * 100, 4) for i in row]
        x.add_row(row)
        print(x)


    folder_path = r"\PTB"
    x_train = get_data(os.path.join(folder_path, "x_train.pt"))
    x_test = get_data(os.path.join(folder_path, "x_test.pt"))
    abnormal = get_data(os.path.join(folder_path, "abnormal.pt"))
    batch_size = 32


    model = CNNAnomalyDetector(chs=x_train.shape[-1])
    model = train_model(model, x_train, num_epochs=100, batch_size=batch_size, learning_rate=1e-4)

    threshold = calculate_threshold(model, x_train, batch_size=batch_size, q = 0.98)

    normal_results = detect_anomalies(model, x_test, threshold, batch_size=batch_size)
    abnormal_results = detect_anomalies(model, abnormal, threshold, batch_size=batch_size)

    print(f"Normal Data Detected as Anomalies: {normal_results.sum().item()} / {len(normal_results)}")
    print(f"Anomalous Data Detected as Anomalies: {abnormal_results.sum().item()} / {len(abnormal_results)}")
    print_table_result(abnormal_sum = len(abnormal_results), normal_sum = len(normal_results),
                       abnormal_true = abnormal_results.sum().item(), normal_true = normal_results.sum().item())



