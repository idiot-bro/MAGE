import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

def auc_roc(y_trues, errors, fontfamily, fontsize):
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    datasets = ["TUSZ", "CHB-MIT", "AUBMC", "INCART", "PTB"]
    plt.figure(figsize=(6, 5))
    for i in range(len(y_trues)):
        fpr, tpr, _ = roc_curve(y_trues[i], errors[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors[i],
                 label=f'{datasets[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontfamily=fontfamily, fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontfamily=fontfamily, fontsize=fontsize)
    plt.title('ROC Curves', fontfamily=fontfamily, fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize, prop={'family': fontfamily})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def pr(y_trues, errors, fontfamily, fontsize):
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    datasets = ["TUSZ", "CHB-MIT", "AUBMC", "INCART", "PTB"]

    plt.figure(figsize=(6, 5))

    for i in range(5):
        precision, recall, _ = precision_recall_curve(y_trues[i], errors[i])
        ap_score = average_precision_score(y_trues[i], errors[i])
        plt.plot(recall, precision, lw=2, color=colors[i], label=f'{datasets[i]} (AUC = {ap_score:.2f})')

    plt.xlabel('Recall', fontfamily=fontfamily, fontsize=fontsize)
    plt.ylabel('Precision', fontfamily=fontfamily, fontsize=fontsize)
    plt.title('Precision-Recall Curves', fontfamily=fontfamily, fontsize=fontsize)
    plt.legend(loc='lower left', fontsize=fontsize, prop={'family': fontfamily})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def auc_roc_pr(y_trues, errors, fontfamily, fontsize):
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    datasets = ["TUSZ", "CHB-MIT", "AUBMC", "INCART", "PTB"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(5):
        fpr, tpr, _ = roc_curve(y_trues[i], errors[i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2, color=colors[i], label=f'{datasets[i]}(AUC = {roc_auc:.4f})')

    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机模型
    axes[0].set_title('(a) ROC Curves', fontfamily=fontfamily, fontsize=16)
    axes[0].set_xlabel('False Positive Rate', fontfamily=fontfamily, fontsize=fontsize)
    axes[0].set_ylabel('True Positive Rate', fontfamily=fontfamily, fontsize=fontsize)
    axes[0].legend(loc='lower right', fontsize=12)
    axes[0].grid(alpha=0.3)

    for i in range(5):
        precision, recall, _ = precision_recall_curve(y_trues[i], errors[i])
        ap_score = average_precision_score(y_trues[i], errors[i])
        axes[1].plot(recall, precision, lw=2, color=colors[i], label=f'{datasets[i]} (AUC = {ap_score:.4f})')

    axes[1].set_title('(b) Precision-Recall Curves', fontfamily=fontfamily, fontsize=16)
    axes[1].set_xlabel('Recall', fontfamily=fontfamily, fontsize=fontsize)
    axes[1].set_ylabel('Precision', fontfamily=fontfamily, fontsize=fontsize)
    axes[1].legend(loc='lower left', fontsize=13)
    axes[1].grid(alpha=0.3)

    # 图像整体调整
    plt.tight_layout()
    plt.savefig(r'figures/auc-roc&pr.png', bbox_inches='tight', dpi=500)
    plt.show()

def readpkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # datasets = ["TUSZ", "CHB-MIT", "AUBMC", "INCART", "PTB"]

    tusz_test = readpkl(os.path.join('raw-data', 'loss', 'tusz-test-loss.pkl'))
    chbmit_test = readpkl(os.path.join('raw-data', 'loss', 'chbmit-test-loss.pkl'))
    aubmc_test = readpkl(os.path.join('raw-data', 'loss', 'aubmc-test-loss.pkl'))
    incart_test = readpkl(os.path.join('raw-data', 'loss', 'incart-test-loss.pkl'))
    ptb_test = readpkl(os.path.join('raw-data', 'loss', 'ptb-test-loss.pkl'))

    tusz_abnormal = readpkl(os.path.join('raw-data', 'loss', 'tusz-abnormal-loss.pkl'))
    chbmit_abnormal = readpkl(os.path.join('raw-data', 'loss', 'chbmit-abnormal-loss.pkl'))
    aubmc_abnormal = readpkl(os.path.join('raw-data', 'loss', 'aubmc-abnormal-loss.pkl'))
    incart_abnormal = readpkl(os.path.join('raw-data', 'loss', 'incart-abnormal-loss.pkl'))
    ptb_abnormal = readpkl(os.path.join('raw-data', 'loss', 'ptb-abnormal-loss.pkl'))
    y_trues = [
        np.array([0] * len(tusz_test['test mse loss']) + [1] * len(tusz_abnormal['test mse loss'])),
        np.array([0] * len(chbmit_test['test mse loss']) + [1] * len(chbmit_abnormal['test mse loss'])),
        np.array([0] * len(aubmc_test['test mse loss']) + [1] * len(aubmc_abnormal['test mse loss'])),
        np.array([0] * len(incart_test['test mse loss']) + [1] * len(incart_abnormal['test mse loss'])),
        np.array([0] * len(ptb_test['test mse loss']) + [1] * len(ptb_abnormal['test mse loss']))
    ]
    errors = [
        np.array(tusz_test['test mse loss'] + tusz_abnormal['test mse loss']),
        np.array(chbmit_test['test mse loss'] + chbmit_abnormal['test mse loss']),
        np.array(aubmc_test['test mse loss'] + aubmc_abnormal['test mse loss']),
        np.array(incart_test['test mse loss'] + incart_abnormal['test mse loss']),
        np.array(ptb_test['test mse loss'] + ptb_abnormal['test mse loss'])
    ]

    # mse_loss_list = [tusz['train mse loss'], chbmit['train mse loss'], aubmc['train mse loss'],
    #                  incart['train mse loss'], ptb['train mse loss']]




    auc_roc_pr(y_trues = y_trues, errors = errors,fontfamily = 'Arial',fontsize=14)