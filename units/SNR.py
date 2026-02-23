import numpy as np

def add_noise_by_snr(signal, snr_db):

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear


    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)


    noisy_signal = signal + noise

    return noisy_signal

if __name__ == '__main__':
    eeg = np.random.randn(17, 384)  # 假设一段 EEG 信号（3秒、128Hz）
    eeg_snr_0 = add_noise_by_snr(eeg, 0)
    eeg_snr_5 = add_noise_by_snr(eeg, 5)
    eeg_snr_10 = add_noise_by_snr(eeg, 10)

