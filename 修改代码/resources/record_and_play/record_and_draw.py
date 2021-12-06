'''录制文件并绘制时频域

自动从8通道截取有效6通道
'''
from numpy.core.defchararray import array
import pyaudio
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyheatmap.heatmap import HeatMap
import createMic as Mic
import time
import cv2
from numpy import *
import mic_array_api as mic


if __name__ == '__main__':
    mics = mic.mic_array()

    RESPEAKER_RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 0.02
    data = mics.get_data(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)

    fft_data = abs(np.fft.fft(data.T))

    # yuantu
    plt.figure()
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.plot(data[:, i-1])
    plt.savefig('./data.jpg')

    # fft tu
    plt.figure()
    length = 24000
    new_ticks = np.arange(0, length, length/511)

    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.plot(new_ticks, fft_data[i-1, 0:int(fft_data.shape[1]/2)-1])
    plt.savefig('./fft_data.jpg')

    plt.show()
    mics.close()
