import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 计算ROC曲线
def evaluate_roc(errors, labels):
    fpr, tpr, thresholds = roc_curve(labels, errors)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return auc_score

if __name__ == '__main__':
    pass
