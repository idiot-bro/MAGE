import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
def roccurve(labels, reconstruction_errors, save_path=None):

    reconstruction_errors = reconstruction_errors / np.max(reconstruction_errors)

    fpr, tpr, thresholds = roc_curve(labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'ROC Curve.png'), dpi= 500)
    plt.show()

if __name__ == '__main__':
    pass
