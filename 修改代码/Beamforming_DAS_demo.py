import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from preprocess.developCSM import developCSM, freqs_precaulate
from preprocess.steerVector import steerVector
from Algorithm.DAS import DAS
from preprocess.simulateArraydata import simulateArraydata
import wave
import scipy.io as scio
import time


def get_micSignal_from_wav(wav_path):
    '''从wav文件中获得micsignal
    自动进行裁切(8通道到6通道)
    mic_sigal.size=channel,frames_total'''
    wavfile = wave.open(wav_path)
    framerate = wavfile.getframerate()
    params = wavfile.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = wavfile.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    # waveData = waveData*1.0/(max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nchannels, nframes])
    wavfile.close()

    mic_signal = TrackAlignment(waveData)
    mic_signal = mic_signal.T  # mic_sigal.size=channel,frames_total
    return framerate, nframes, mic_signal


def TrackAlignment(data):
    '''裁切'''
    y = data.T
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


def simulateMicsignal(mic_pos, z_source, c, framerate, mic_centre, t_start, t_end):
    '''构建虚拟声源点'''
    # % 构建声源点  %注:设定信号持续时间和整合声源信息：source_info
    source_x = np.array([-0.2, 0.3]).reshape(1, -1).T  # source_x = [-1,0.5]';
    source_y = np.array([0, 0.1]).reshape(1, -1).T  # source_y = [0,1]';

    # % 设定声源频率
    source1_freq = 2000
    source2_freq = 3000  # source1_freq = 2000;  source2_freq = 3000
    # sources_freq = [source1_freq, source2_freq]';  %注:整合声源信息：source_info
    sources_freq = np.array([source1_freq, source2_freq]).reshape(1, -1).T

    # % 设定信号持续时间
    # source_duration = t_end*ones(length(source_x), 1)
    # %注: 整合声源信息：source_info和获取麦克风阵列输出simulateArraydata
    source_duration = t_end*np.ones((max(source_x.shape), 1))

    # % 设定声源声压有效值
    source1_spl = 100
    source2_spl = 100  # source1_spl = 100; source2_spl = 100
    # sources_spl = [source1_spl, source2_spl].';   %注:整合声源信息：source_info
    sources_spl = np.array([source1_spl, source2_spl]).reshape(1, -1).T

    # % 整合声源信息：source_info % 注: 获取麦克风阵列输出simulateArraydata
    # % 声源点坐标x / 声源点坐标y / 声源点坐标z（到扫描平面距离）/ 声源频率 / 声压值

    # source_info = [source_x, source_y, z_source *
    #                ones(length(source_x), 1), sources_freq, sources_spl, source_duration]
    source_info = np.concatenate((source_x, source_y, z_source*np.ones(
        (max(source_x.shape), 1)), sources_freq, sources_spl, source_duration), axis=1)
    # https://www.cnblogs.com/cymwill/p/8358866.html

    # % 获取麦克风阵列输出 % 注:计算CSM以及确定扫描频率developCSM
    mic_signal = simulateArraydata(
        source_info, mic_pos, c, framerate, source_duration, mic_centre)
    # mic_signal.size() = (nchannels, nframes) 且不归一化
    return mic_signal


def get_MicSignal():
    return


def save_wav():
    return


def get_micArray(path_full):
    try:
        darray = scio.loadmat(path_full)
    except:
        darray = h5py.File(path_full)  # 如果python 报错
    array = darray['array'][:]
    mic_x_axis = array[:, 0]
    mic_y_axis = array[:, 1]
    mic_z_axis = 0
    mic_pos = np.transpose([mic_x_axis, mic_y_axis])
    mic_pos = np.concatenate(
        (mic_pos, np.ones((mic_x_axis.size, 1))*mic_z_axis), axis=1)
    # mic_centre = mean(mic_pos); % 阵列中心的坐标
    mic_centre = mic_pos.mean(axis=0).reshape(1, -1)
    return mic_pos, mic_centre, mic_x_axis, mic_y_axis


def draw_mic_array(mic_x_axis, mic_y_axis):
    '''绘制麦克风阵列'''
    plt.figure()
    plt.plot(mic_x_axis, mic_y_axis, 'k.', markersize=20)
    plt.xlim([min(mic_x_axis)-0.1, max(mic_x_axis)+0.1])
    plt.ylim([min(mic_y_axis)-0.1, max(mic_y_axis)+0.1])
    plt.show()


def plot_figure(X, Y, SPL):
    # % 绘制波束成像图

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()  # figure;
    BF_dr = 6
    # BF_dr = 6; maxSPL = ceil(max(SPL(:)))
    maxSPL = np.ceil(max(SPL.flatten('F')))
    X, Y = np.meshgrid(X, Y)
    plt.contourf(X, Y, SPL, np.arange((maxSPL-BF_dr), maxSPL+1, 1))
    plt.colorbar()
    plt.xlabel('x轴(m)')
    plt.ylabel('y轴(m)')
    plt.title('波束成像图')
    plt.show()
