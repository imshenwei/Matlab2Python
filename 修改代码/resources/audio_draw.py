# -*- coding: utf-8 -*-
"""wav文件绘图功能

自动根据读入的wav文件进行音频绘制
"""

import wave
import matplotlib.pyplot as plt
import numpy as np
import os

# filepath = "Ex_1031_/"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
f = wave.open("output.wav", 'rb')

# 获取音频信息
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
# waveData = waveData*1.0/(max(abs(waveData)))  # * wave幅值归一化
waveData = np.reshape(waveData, [nframes, nchannels])
f.close()

time = np.arange(0, nframes)*(1.0 / framerate)  # plot the wave

# 绘图
for i in range(1, nchannels):
    plt.figure()
    plt.plot(time, waveData[:, i])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Ch-"+str(i)+" wavedata")
    plt.grid('on')  # 标尺，on：有，off:无。
plt.show()
