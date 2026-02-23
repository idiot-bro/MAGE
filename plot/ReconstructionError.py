import matplotlib.pyplot as plt
import os
def reconstruction_error(train_mse_loss, eval_mse_loss, test_mse_loss, threshold, save_path=None):

    plt.hist(train_mse_loss, bins=30, alpha=0.6, label='Train Errors')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {round(threshold, 5)}')
    if eval_mse_loss is not None:
        plt.hist(eval_mse_loss, bins=30, alpha=0.6, label='Eval Errors')
    plt.hist(test_mse_loss, bins=30, alpha=0.6, label='Test Errors')
    plt.legend()
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'Reconstruction Error Distribution.png'), dpi= 500)
    plt.show()



if __name__ == '__main__':
    pass
