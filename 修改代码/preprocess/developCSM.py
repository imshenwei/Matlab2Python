# import math
# import wave as we

# import matplotlib.pyplot as plt
import numpy as np
from numpy import *  # noqa


def developCSM(mic_signal, freq_l, freq_u, Fs, t_start, t_end):
    # 生成互谱矩阵 CSM

    # 麦克风阵列数
    N_mic = size(mic_signal, 1)
    # 开始和结束的样本点
    start_sample = floor(t_start*Fs)+1
    end_samples = ceil(t_end * Fs)
    # 选取在扫描频率之间的点
    # x_fr = Fs / end_samples * (0:floor(end_samples/2)-1);
    x_fr = Fs / end_samples * \
        np.arange(0, floor(end_samples/2)-1+1).reshape(1, -1)
    # freq_sels = find((x_fr>=search_freql).*(x_fr<=search_frequ));
    freq_sels = np.where((x_fr >= freq_l)*(x_fr <= freq_u))
    freq_sels = freq_sels[1].reshape(1, -1)
    # 扫描频点的个数
    N_freqs = max(freq_sels.shape)
    # 初始化互谱矩阵CSM
    CSM = np.array(np.zeros((N_mic, N_mic, N_freqs)), dtype=complex)
    # 对采集到的时域数据进行傅里叶变换

    mic_signal_fft = sqrt(2) * \
        np.fft.fft(mic_signal[np.arange(
            start_sample-1, (end_samples + 1)-1, 1, dtype=int), :].T)/(end_samples-start_sample+1)  # mic_signal_fft = sqrt(2)*fft(mic_signal(start_sample:end_samples,:))/(end_samples-start_sample+1);
    # https://www.zhihu.com/question/433589063/answer/1615980074
    # https://blog.csdn.net/tengqi200/article/details/117334718

    # 生成互谱矩阵 CSM
    mic_signal_fft = np.mat(mic_signal_fft.T)
    for F in np.arange(0, N_freqs, 1):
        # 计算第K个频率下的互谱矩阵
        CSM[:, :, F] = CSM[:, :, F] + \
            np.dot(mic_signal_fft[freq_sels[0][F], :].T,
                   mic_signal_fft[freq_sels[0][F], :].conjugate())

        # CSM[:, :, F] = mic_signal_fft[freq_sels[0][F],
        #                               :].T @ mic_signal_fft[freq_sels[0][F], :].conjugate()
        freqs = x_fr[0, [freq_sels[0]]]
    return [CSM, freqs]
