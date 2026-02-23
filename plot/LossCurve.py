import matplotlib.pyplot as plt
import os
def loss_curve(train_losses, eval_losses = None, save_path = None):

    plt.plot(train_losses, label='Training Loss')
    if eval_losses is not None:
        if len(eval_losses) != 0:
            plt.plot(eval_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'Loss Curve.png'), dpi= 500)
    plt.show()

if __name__ == '__main__':
    loss_curve(train_losses=[0,1,2,3], eval_losses=[])
