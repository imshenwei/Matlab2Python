import numpy as np
from numpy import *
import wave as we
import matplotlib.pyplot as plt


def developCSM(mic_signal, freq_l, freq_u, Fs, t_start, t_end):
    # 生成互谱矩阵 CSM

    # 麦克风阵列数
    N_mic = size(mic_signal, 1)

# 开始和结束的样本点
    start_sample = floor(t_start*Fs)+1
    end_samples = ceil(t_end * Fs)

# 选取在扫描频率之间的点
    x_fr = Fs / end_samples * np.arange(0, int(floor(end_samples/2)-1), 1)
    freq_sels = np.where((x_fr >= freq_l)*(x_fr <= freq_u))

# 扫描频点的个数
    N_freqs = len(freq_sels[0])

# 初始化互谱矩阵CSM
    CSM = np.array(np.zeros((N_mic, N_mic, N_freqs)), dtype=complex)

# 对采集到的时域数据进行傅里叶变换
    mic_signal_fft = sqrt(2) * \
        np.fft.fft(mic_signal[np.arange(
            start_sample, (end_samples + 1), 1, dtype=int), :].T)/(end_samples-start_sample+1)


# 生成互谱矩阵 CSM
    mic_signal_fft = np.mat(mic_signal_fft.T)
    for F in np.arange(0, N_freqs, 1):
        # 计算第K个频率下的互谱矩阵
        CSM[:, :, F] = CSM[:, :, F] + \
            np.dot(mic_signal_fft[freq_sels[0][F], :].T,
                   mic_signal_fft[freq_sels[0][F], :].conjugate())

        # CSM[:, :, F] = mic_signal_fft[freq_sels[0][F],
        #                               :].T @ mic_signal_fft[freq_sels[0][F], :].conjugate()
        freqs = x_fr[freq_sels[0]]
    return [CSM, freqs]


def wavread(path):
    wavfile = we.open(path, "rb")
    params = wavfile.getparams()
    framesra, frameswav = params[2], params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav, dtype=np.int16)
    datause.shape = -1, 8
    datause = datause.T
    time = np.arange(0, frameswav) * (1.0/framesra)
    return datause, time


if __name__ == '__main__':
    wavdata, _ = wavread('output.wav')
    wavdata = wavdata
    plt.figure()
    for i in range(1, 9):
        plt.subplot(2, 4, i)
        plt.plot(wavdata[i-1, 4000:])
    plt.show()

    mic_signal = wavdata[0: 6, 4000:5000]

    Fs = 48000
    freq_l = 800
    freq_u = 4000
    t_start = 0
    t_end = 0.02

    [CSM, freqs] = developCSM(mic_signal.T, freq_l, freq_u, Fs, t_start, t_end)

    print('a')
