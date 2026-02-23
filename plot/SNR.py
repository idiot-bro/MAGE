import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
def add_gaussian_noise_snr(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def plot_eeg_snrs_subplots(noisy_signals:dict, snr_levels=None,fontsize=None, fontfamily=None,):

    n_subplots = len(snr_levels)
    n_cols = 2
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()


    for i, snr in enumerate(snr_levels):
        ax = axes[i]
        if snr == 'Raw':
            title = 'Raw'
        else:
            title = f'SNR {snr} dB'

        signal = noisy_signals[title]
        ax.plot(signal, color='black', linewidth=1.2)
        ax.set_title(title, fontsize=16, fontfamily=fontfamily)
        ax.set_xlabel('Time Points', fontsize=16, fontfamily=fontfamily)
        ax.set_ylabel('Amplitude', fontsize=16, fontfamily=fontfamily)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        ax.tick_params(axis='both', labelsize=14)

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # plt.savefig(rf'figures/SNR-eeg ({fontfamily}).png', bbox_inches='tight', dpi=500)
    plt.show()


if __name__ == '__main__':

    data = np.load(r"normal.npy")[0]
    snr_levels = ['Raw', -5, 0, 5, 10, 15, 20]
    noisy_signals = {}

    channel = 0
    noisy_signals['Raw'] = data[channel][:128]
    print(noisy_signals['Raw'].shape)
    for snr in snr_levels[1:]:
        noisy_signals[f'SNR {snr} dB'] = add_gaussian_noise_snr(data[channel][:128], snr)

    with open(r"raw-data/SNR-eeg.pkl", "wb") as f:
        pickle.dump(noisy_signals, f)


    plot_eeg_snrs_subplots(noisy_signals, snr_levels=snr_levels, fontsize=16, fontfamily='Arial',)

