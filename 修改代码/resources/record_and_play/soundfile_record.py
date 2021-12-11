import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def TrackAlignment(data):
    y = data
    s_x = np.argsort(np.max(y, axis=0))
    y2 = []
    if abs(s_x[0]-s_x[1]) == 1:
        s_x_max = max(s_x[0:1])
        if(s_x_max == 1):
            y2 = y[:, 2:8]
        elif(s_x_max < 7 and s_x_max > 1):
            y2 = np.hstack((y[:, s_x_max+1:8], y[:, 0:s_x_max-1]))
        else:
            y2 = y[:, 0:6]
    else:
        y2 = y[:, 1:7]
    return y2


def draw(data):
    fft_data = abs(np.fft.fft(data.T))

    # yuantu
    plt.figure()
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.plot(data[:, i-1])
    # plt.savefig('./data.jpg')

    # fft tu
    plt.figure()
    length = 24000
    new_ticks = np.arange(0, length, length/(int(fft_data.shape[1]/2)-1))

    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.plot(new_ticks, fft_data[i-1, 0:int(fft_data.shape[1]/2)-1])
    # plt.savefig('./fft_data.jpg')

    # plt.show()


def save_micData(samplerate=48000, duration=0.02):  # Hertz
    '''从麦克风文件中获得micsignal
    自动进行裁切(8通道到6通道)
    mic_sigal.size=channel,frames_total
    IN: 采样频率sameplerate, 采样持续时间duration
    OUT: 采样频率framerate, 采样点总数nframes, 6通道麦克风信号mic_signal,采样持续时间duration'''
    # seconds
    # filename = 'output1.wav'

    mydata = sd.rec(int(samplerate * duration), samplerate=48000,
                    channels=8, blocking=True)
    mydata = TrackAlignment(mydata)
    # draw(mydata)
    return samplerate, int(samplerate * duration), mydata, duration

    #sf.write(filename, mydata, samplerate)


# save_micData()
