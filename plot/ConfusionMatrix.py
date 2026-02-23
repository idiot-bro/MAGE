import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np

def confusionmatrix(true_label, predicted_label, threshold, save_path=None):

    cm = confusion_matrix(true_label, predicted_label)


    conf_matrix_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=["Normal", "Anomaly"])
    # 绘制并显示混淆矩阵
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")  # values_format=".2f"

    plt.title(f"Confusion Matrix (Threshold = {round(threshold, 5)})")
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "Confusion Matrix.png"), dpi= 500)
    plt.show()
    plt.close()
if __name__ == '__main__':
    pass
