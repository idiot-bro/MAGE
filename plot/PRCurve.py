import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

def pr_curve(labels, reconstruction_errors, save_path = None):

    reconstruction_errors = reconstruction_errors / np.max(reconstruction_errors)

    precision, recall, thresholds = precision_recall_curve(labels, reconstruction_errors)
    average_precision = average_precision_score(labels, reconstruction_errors)

    plt.plot(recall, precision, color='b', label='Precision-Recall curve (AP = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "Precision Recall Curve.png"), dpi= 500)
    plt.show()

if __name__ == '__main__':
    pass
