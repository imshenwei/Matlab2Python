import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from preprocess.developCSM import developCSM
from preprocess.steerVector import steerVector
from Algorithm.DAS import DAS
from preprocess.simulateArraydata import simulateArraydata
import wave
import scipy.io as scio
import time
from resources.record_and_play.record import save_wav_only8
from resources.record_and_play.pyaudio_record import save_wav_2and8
import resources.record_and_play.mic_array_api as mic
from resources.record_and_play.soundfile_record import save_micData


def beamforming():
    '''DAS 波束成像算法（扫频模式）Delay Summation Algorithm'''

    z_source = 1  # 麦克风阵列平面与扫描屏幕的距离

    framerate = 48000  # 麦克风采样频率
    # 确定扫描频段（800-4000 Hz）
    search_freql = 800
    search_frequ = 4000
    mic_r = 0.5  # 麦克风阵列限定半径
    c = 343  # 声速
    scan_resolution = z_source/20  # 0.05#0.1  # 扫描网格的分辨率
    scan_r = z_source/2  # 扫描声源限定半径

    #  信号的采样频率
    # 引用: https://www.cnblogs.com/xingshansi/p/6799994.html

    # 麦克风阵列限定区域
    mic_x = np.array([-mic_r, mic_r])
    mic_y = np.array([-mic_r, mic_r])
    # 扫描声源限定区域
    scan_x = np.array([-scan_r, scan_r])
    scan_y = np.array([-scan_r, scan_r])

    # try:
    #     save_wav_only8()
    # except:
    #     save_wav_2and8()
    # wav_path = "/home/pi/Desktop/修改代码/resources/output.wav"
    # framerate, nframes, mic_signal = get_micSignal_from_wav(wav_path)
    framerate, nframes, mic_signal, duration = save_micData()
    # 导入麦克风阵列
    path_full = '/home/pi/Desktop/修改代码/resources/6_spiral_array.mat'  # 须要读取的mat文件路径
    # path_full = '修改代码/resources/2_spiral_array.mat'
    mic_pos, mic_centre, mic_x_axis, mic_y_axis = get_micArray(path_full)

    # draw_mic_array(mic_x_axis, mic_y_axis)

    # 设定信号持续时间
    t_start = 0
    t_end = duration

    # mic_signal = simulateMicsignal(source_info, mic_info, c, fs, duration, mic_centre)

    time_start_total = time.time()
    time_start = time.time()
    [CSM, freqs] = developCSM(mic_signal.T, search_freql,
                              search_frequ, framerate, t_start, t_end)
    time_end = time.time()
    print('csm cost', time_end-time_start)

    time_start = time.time()
    g = steerVector(z_source, freqs, [scan_x, scan_y],
                    scan_resolution, mic_pos.T, c, mic_centre)
    time_end = time.time()
    print('steervector cost', time_end-time_start)

    # 波束成像 -- DAS算法
    time_start = time.time()
    [X, Y, B] = DAS(CSM, g, freqs, [scan_x, scan_y], scan_resolution)
    time_end = time.time()
    print('DAS cost', time_end-time_start)

    # % 声压级单位转换
    B[B < 0] = 0
    eps = np.finfo(np.float64).eps
    SPL = 20*np.log10((eps+np.sqrt(B.real))/2e-5)

    time_end_total = time.time()
    # plot_figure(X, Y, SPL)
    print('totally cost', time_end_total-time_start_total)
    return SPL


def simulateMicsignal():
    '''构建虚拟声源点'''
    # % 构建声源点  %注:设定信号持续时间和整合声源信息：source_info
    source_x = np.array([-1, 0.5]).reshape(1, -1).T  # source_x = [-1,0.5]';
    source_y = np.array([0, 1]).reshape(1, -1).T  # source_y = [0,1]';

    # % 设定声源频率
    source1_freq = 2000
    source2_freq = 3000  # source1_freq = 2000;  source2_freq = 3000
    # sources_freq = [source1_freq, source2_freq]';  %注:整合声源信息：source_info
    sources_freq = np.array([source1_freq, source2_freq]).reshape(1, -1).T

    # % 设定信号持续时间  %计算CSM以及确定扫描频率developCSM
    t_start = 0
    t_end = nframes/framerate  # t_start = 0;  t_end = 0.02
    source_duration = t_end*ones(length(source_x), 1)
    # %注: 整合声源信息：source_info和获取麦克风阵列输出simulateArraydata
    source_duration = t_end*np.ones((max(source_x.shape), 1))

    # % 设定声源声压有效值
    source1_spl = 100
    source2_spl = 100  # source1_spl = 100; source2_spl = 100
    # sources_spl = [source1_spl, source2_spl].';   %注:整合声源信息：source_info
    sources_spl = np.array([source1_spl, source2_spl]).reshape(1, -1).T

    # % 整合声源信息：source_info % 注: 获取麦克风阵列输出simulateArraydata
    # % 声源点坐标x / 声源点坐标y / 声源点坐标z（到扫描平面距离）/ 声源频率 / 声压值

    source_info = [source_x, source_y, z_source *
                   ones(length(source_x), 1), sources_freq, sources_spl, source_duration]
    source_info = np.concatenate((source_x, source_y, z_source*np.ones(
        (max(source_x.shape), 1)), sources_freq, sources_spl, source_duration), axis=1)
    # https://www.cnblogs.com/cymwill/p/8358866.html

    # % 获取麦克风阵列输出 % 注:计算CSM以及确定扫描频率developCSM
    mic_signal = simulateArraydata(
        source_info, mic_pos, c, framerate, source_duration, mic_centre)
    mic_signal = simulateArraydata(
        source_info, mic_pos, c, framerate, source_duration, mic_centre)
    # mic_signal.size() = (nchannels, nframes) 且不归一化
    return mic_signal


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


def get_micSignal_from_wav(wav_path):
    '''从wav文件中获得micsignal
    自动进行裁切(8通道到6通道)
    mic_sigal.size=channel,frames_total
    IN: wav存储路径wav_path
    OUT: 采样频率framerate, 采样点总数nframes, 6通道麦克风信号mic_signal'''
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
