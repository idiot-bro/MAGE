import os
import torch
from tqdm import tqdm
import sys
import torch.optim as optim
import numpy as np
from prettytable import PrettyTable
import units.check_cuda
import pickle
from model.MAGE import MAGE
from model.AMSL import AMSL

from plot.ConfusionMatrix import confusionmatrix
from plot.LossCurve import loss_curve
from plot.PRCurve import pr_curve
from plot.ROCCurve import roccurve
from plot.ReconstructionError import reconstruction_error
from units.construct_dataset import CustomDataset, CustomDataloader
from units.load_configuration import load_configuration
from units.adversarial import fgsm_attack, pgd_attack, cw_attack, tf_attack
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(folder_path, batch_size, device = 'cpu', cuda_id = None, train_file = 'train.pt', test_file= 'test.pt', abnormal_file= 'abnormal.pt',
              shuffle = True, pin_memory = True, collate_fn = None, num_workers = 0):
    train_dataset = CustomDataset(file_path = os.path.join(folder_path, train_file), device=device, cuda_id = cuda_id)
    test_dataset = CustomDataset(file_path = os.path.join(folder_path, test_file), device=device, cuda_id = cuda_id)
    abnormal_dataset = CustomDataset(file_path = os.path.join(folder_path, abnormal_file), device=device, cuda_id = cuda_id)
    train_dataloader = CustomDataloader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = pin_memory, collate_fn = collate_fn, num_workers=num_workers)
    test_dataloader = CustomDataloader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = pin_memory, collate_fn = collate_fn, num_workers=num_workers)
    abnormal_dataloader = CustomDataloader(dataset=abnormal_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = pin_memory, collate_fn = collate_fn, num_workers=num_workers)
    return train_dataloader, test_dataloader, abnormal_dataloader
def initialize_model(config, device = 'cpu'):
    model = MAGE(filter_size0 = config['filter_size0'], filter_size1 = config['filter_size1'], filter_size2=config['filter_size2'] , filter_size3 = config['filter_size3'],
                 trans=config['trans'], dropout=config['dropout'], ss_output_features = config['ss_output_features'], global_mem_dim = config['global_mem_dim'],
                 heads=config['heads'], temperature=config['temperature'], MultiHead=config['MultiHead'],
                 local_mem_dim = config['local_mem_dim'], fc_dim = config['fc_dim'], momentum = config['momentum'], lambda1 = config['lambda1'], lambda2 = config['lambda2'],
                 ss_kernel_size = config['ss_kernel_size'], decoder_kernel_sizes = config['decoder_kernel_size'], strides = config['decoder_stride'], paddings = config['decoder_padding'],
                 device=f"cuda:{config['cuda_id']}" if config['cuda_id'] is not None else "cpu")
    if device != 'cpu' and config['cuda_id'] is not None:
        print(f"Using GPU {config['cuda_id']}")
        model = model.to(torch.device(f"cuda:{config['cuda_id']}"))
    return model

def test(model, test_loader):
    epoch_dict = {'test mse loss': [],
                  'test sparse loss': [],
                  'test classification loss': [],
                  'test total loss': []}
    model.eval()
    with torch.no_grad():
        for _, input_data in enumerate(test_loader):
            _, mse_loss, sparse_loss, loss_g, total_loss = model(input_data)
            epoch_dict['test mse loss'].append(mse_loss.item())
            epoch_dict['test sparse loss'].append(sparse_loss.item())
            epoch_dict['test classification loss'].append(loss_g.item())
            epoch_dict['test total loss'].append(total_loss.item())
    return epoch_dict

def train(model, config, optimizer, train_dataloader, test_dataloader = None,
          disable_progress =False, is_adversarial = False, adversarial = 'FGSM',
          lambda_adv = 0.5, decimal_point = 8, abnormal_dataloader = None):
    epochs_dict = {'train mse loss':[],
                   'train sparse loss':[],
                   'train classification loss':[],
                   'train total loss': [],
                   'test mse loss':[],
                   'test sparse loss':[],
                   'test classification loss':[],
                   'test total loss': []}
    model.train()
    for epoch in range(config['epochs']):
        model.train()
        epoch_dict = {'train mse loss': [],
                       'train sparse loss': [],
                       'train classification loss': [],
                       'train total loss': []}
        for _, input_data in enumerate(tqdm(train_dataloader, file=sys.stdout, desc='Training', unit='epoch', disable=disable_progress)):
            optimizer.zero_grad()
            _, mse_loss, sparse_loss, loss_g, total_loss = model(input_data)
            epoch_dict['train mse loss'].append(mse_loss.item())
            epoch_dict['train sparse loss'].append(sparse_loss.item())
            epoch_dict['train classification loss'].append(loss_g.item())
            epoch_dict['train total loss'].append(total_loss.item())
            if is_adversarial:
                if adversarial == 'FGSM':
                    adv_input_data = fgsm_attack(model, x = input_data, epsilon=0.1)
                elif adversarial == 'PGD':
                    adv_input_data = pgd_attack(model, x=input_data, epsilon=0.1, alpha=0.01, num_iter=10)
                elif adversarial == 'CW':
                    adv_input_data = cw_attack(model, x=input_data, lr=0.01, num_iter=10, c=1)
                elif adversarial == 'TF':
                    adv_input_data = tf_attack(model, x=input_data, lr=0.001, num_iter=10)
                _, _, _, _, adv_total_loss = model(adv_input_data)
                loss = total_loss + lambda_adv * adv_total_loss
            else:
                loss = total_loss
            loss.backward()
            optimizer.step()

        epochs_dict['train mse loss'].append(sum(epoch_dict['train mse loss']) / len(epoch_dict['train mse loss']))
        epochs_dict['train sparse loss'].append(sum(epoch_dict['train sparse loss']) / len(epoch_dict['train sparse loss']))
        epochs_dict['train classification loss'].append(sum(epoch_dict['train classification loss']) / len(epoch_dict['train classification loss']))
        epochs_dict['train total loss'].append(sum(epoch_dict['train total loss']) / len(epoch_dict['train total loss']))
        # log = 'Epoch [{}/{}], Train Loss: {:.8f} '.format(epoch + 1, config['epochs'], sum(loss_list) / len(loss_list))
        if test_dataloader is not None:
            test_epoch_dict = test(model, test_dataloader)
            epochs_dict['test mse loss'].append(sum(test_epoch_dict['test mse loss']) / len(test_epoch_dict['test mse loss']))
            epochs_dict['test sparse loss'].append(sum(test_epoch_dict['test sparse loss']) / len(test_epoch_dict['test sparse loss']))
            epochs_dict['test classification loss'].append(sum(test_epoch_dict['test classification loss']) / len(test_epoch_dict['test classification loss']))
            epochs_dict['test total loss'].append(sum(test_epoch_dict['test total loss']) / len(test_epoch_dict['test total loss']))
            # epoch_dict.update(test_epoch_dict)

        x = PrettyTable(["Train total loss", "Train mse loss", "Train sparse loss", "Train classification loss",
                         "Test total loss", "Test mse loss", "Test sparse loss", "Test classification loss"])
        x.add_row([round(epochs_dict['train total loss'][epoch], decimal_point), round(epochs_dict['train mse loss'][epoch], decimal_point),
                   round(epochs_dict['train sparse loss'][epoch], decimal_point), round(epochs_dict['train classification loss'][epoch], decimal_point),
                   round(epochs_dict['test total loss'][epoch], decimal_point), round(epochs_dict['test mse loss'][epoch], decimal_point),
                   round(epochs_dict['test sparse loss'][epoch], decimal_point), round(epochs_dict['test classification loss'][epoch], decimal_point)])
        print('Epoch [{}/{}].'.format(epoch + 1, config['epochs']))
        print(x)

        #############
        if abnormal_dataloader is not None:
            threshold, train_loss_dict, test_loss_dict, abnormal_loss_dict = (
                set_threshold(model, train_dataloader, test_dataloader, abnormal_dataloader, q=config['q']))
            print_table_result(threshold, test_loss_dict, abnormal_loss_dict)
        #############

    print('Model training completed!')
    return model, epochs_dict
def set_threshold(model, train_dataloader, test_dataloader, abnormal_dataloader, q):
    print("Model testing begins...")
    train_loss_dict = test(model, train_dataloader)
    test_loss_dict = test(model, test_dataloader)
    abnormal_loss_dict = test(model, abnormal_dataloader)
    threshold = torch.quantile(torch.tensor(train_loss_dict['test mse loss']), q).item()
    return threshold, train_loss_dict, test_loss_dict, abnormal_loss_dict

def print_table_result(threshold, test_loss_dict, abnormal_loss_dict):
    print('Start evaluating...')
    test_mse_loss = np.array(test_loss_dict['test mse loss'])
    abnormal_mse_loss = np.array(abnormal_loss_dict['test mse loss'])
    normal_true = test_mse_loss[test_mse_loss[:] < threshold]
    print('Normal_sum :', test_mse_loss.shape[0], 'Normal_true', normal_true.shape[0])
    abnormal_true = abnormal_mse_loss[abnormal_mse_loss[:] >= threshold]
    print('Abnormal_sum :', abnormal_mse_loss.shape[0], 'abnormal_true', abnormal_true.shape[0])
    normal_true = normal_true.shape[0]
    fatigue_true = abnormal_true.shape[0]
    recerr2 = test_mse_loss.shape[0]
    recerr3 = abnormal_mse_loss.shape[0]
    acc = (normal_true + fatigue_true) / (recerr2 + recerr3)
    precision_n = normal_true / (normal_true + recerr3 - fatigue_true)
    precision_a = fatigue_true / (fatigue_true + recerr2 - normal_true)
    pre_avg = (precision_a + precision_n) / 2
    recall_n = normal_true / recerr2
    recall_a = fatigue_true / recerr3
    recall_avg = (recall_a + recall_n) / 2
    F1 = 2 * (pre_avg * recall_avg) / (pre_avg + recall_avg)
    from prettytable import PrettyTable
    x = PrettyTable(["Acc", "Pre_Normal", "Pre_Abnormal", "Pre_avg", "Recall_Normal", "Recall_Abnormal", "Recall_avg", "F1"])
    row = [acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1]
    row = [round(i*100, 4) for i in row]
    x.add_row(row)
    print(x)

def plot():
    pass
def save(model, save_path, result_path, train_loss):
    torch.save(model, os.path.join(save_path, 'trained-model.pth'))
    if not os.path.exists(os.path.join(result_path, 'loss')):
        os.makedirs(os.path.join(result_path, 'loss'))
    with open(os.path.join(result_path, 'loss', 'trained-loss.pkl'), 'wb') as f:
        pickle.dump(train_loss, f)
    print('Model saved.')
if __name__ == '__main__':
    dataset_name = 'AUBMC'
    config = load_configuration(folder_path=os.path.join(os.getcwd(), 'configs'), config_file=dataset_name)
    train_dataloader, test_dataloader, abnormal_dataloader = load_data(config['data_path'], config['batch'], device = device, cuda_id = config['cuda_id'],
                                                                    train_file = 'x_train.pt', test_file= 'x_test.pt', abnormal_file= 'abnormal.pt',
                                                                    shuffle = True, pin_memory = False, collate_fn = None, num_workers = 0)
    model = initialize_model(config, device = device)
    model, train_loss = train(model, config, optimizer=optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=True),
                                        train_dataloader = train_dataloader, test_dataloader = test_dataloader, disable_progress =True,
                                        is_adversarial = config['is_adversarial'], adversarial = config['adversarial'], lambda_adv = config['lambda_adv'],
                                        abnormal_dataloader = abnormal_dataloader)
    threshold, train_loss_dict, test_loss_dict, abnormal_loss_dict = set_threshold(model, train_dataloader, test_dataloader, abnormal_dataloader, q = config['q'])
    print_table_result(threshold, test_loss_dict, abnormal_loss_dict)
    plot()
    save(model, config['model_path'], result_path = config['result_path'], train_loss = train_loss)


