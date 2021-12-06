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


def simulateMicsignal():
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
    '''hello'''
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


def beamforming():
    # % ------ DAS 波束成像算法（扫频模式）Delay Summation Algorithm
    # % ------ 可以设置多个不同频率、不同声压级的点声源
    # % ------ 可以设置不同距离的扫描平面
    # % ------ 可以选择不同的麦克风阵列
    # % ------ 可以设置想要搜索的频段
    # % ------ 可以调整网格分辨率

    # % 添加路径
    # addpath('.\Algorithm')
    # sys.path.append(
    #     r'C:\Users\Admin\Desktop\桌面整理\资料\Gitee备份\computer-backup\XQ3.1\专业\实验\Matlab2Python\修改代码\已完成')
    wav_path = "修改代码/resources/output.wav"

    # % 麦克风阵列限定区域 %注:后面没用
    mic_x = np.array([-0.5, 0.5])  # mic_x = [-0.5 0.5];
    mic_y = np.array([-0.5, 0.5])  # mic_y = [-0.5 0.5];

    # % 扫描声源限定区域   %注: 计算steerVector转向矢量和DAS算法
    scan_x = np.array([-3, 3])  # scan_x = [-3 3];
    scan_y = np.array([-3, 3])  # scan_y = [-3 3];

    # % 麦克风阵列平面与扫描屏幕的距离  % 注:整合声源信息：source_info, 可能是m
    z_source = 1  # z_source = 3;

    # % 声速  %注:获取麦克风阵列输出和steerVector转向矢量
    c = 343  # c = 343;

    # % 信号的采样频率  %注:获取麦克风阵列输出simulateArraydata和计算CSM以及确定扫描频率developCSM
    # 引用: https://www.cnblogs.com/xingshansi/p/6799994.html
    wavfile = wave.open(wav_path)
    framerate = wavfile.getframerate()  # 8000  #framerate = 8000
    params = wavfile.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = wavfile.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    # waveData = waveData*1.0/(max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nchannels, nframes])
    wavfile.close()

    mic_signal = waveData
    mic_signal = TrackAlignment(mic_signal)
    mic_signal = mic_signal.T

    # % 导入麦克风阵列  %注:绘制麦克风阵列
    # load 56_spiral_array.mat
    path_full = '修改代码/resources/6_spiral_array.mat'  # 须要读取的mat文件路径
    darray = scio.loadmat(path_full)
    # darray = h5py.File(path_full) #如果python 报错
    array = darray['array'][:]
    # array = np.array(np.random.rand(2, 2))
    mic_x_axis = array[:, 0]
    mic_y_axis = array[:, 1]
    # mic_x_axis = array(:,1); mic_y_axis = array(:,2); mic_z_axis = 0
    mic_z_axis = 0
    mic_pos = np.transpose([mic_x_axis, mic_y_axis])
    # mic_pos = [mic_x_axis mic_y_axis]; mic_pos(:,3) = mic_z_axis;  %注:获取麦克风阵列输出simulateArraydata和计算转向矢量steerVector
    mic_pos = np.concatenate(
        (mic_pos, np.ones((mic_x_axis.size, 1))*mic_z_axis), axis=1)
    # mic_centre = mean(mic_pos); % 阵列中心的坐标  %注:获取麦克风阵列输出simulateArraydata和计算转向矢量steerVector
    mic_centre = mic_pos.mean(axis=0).reshape(1, -1)

    def draw_mic_array(mic_x_axis, mic_y_axis):
        # 绘制麦克风阵列
        plt.figure()  # figure;
        # plot(mic_x_axis, mic_y_axis, 'k.', 'MarkerSize', 20)
        plt.plot(mic_x_axis, mic_y_axis, 'k.', markersize=20)
        # xlim([min(mic_x_axis)-0.1, max(mic_x_axis)+0.1])
        plt.xlim([min(mic_x_axis)-0.1, max(mic_x_axis)+0.1])
        # ylim([min(mic_y_axis)-0.1, max(mic_y_axis)+0.1])
        plt.ylim([min(mic_y_axis)-0.1, max(mic_y_axis)+0.1])
        plt.show()

    # draw_mic_array(mic_x_axis, mic_y_axis)

    # # % 设定信号持续时间  %计算CSM以及确定扫描频率developCSM
    t_start = 0
    t_end = nframes/framerate  # t_start = 0;  t_end = 0.02

    # mic_signal = simulateMicsignal(source_info, mic_info, c, fs, duration, mic_centre)

    # % 确定扫描频段（800-4000 Hz）% 注:计算CSM以及确定扫描频率developCSM
    search_freql = 800  # search_freql = 800;
    search_frequ = 4000  # search_frequ = 4000;
    time_start_total = time.time()

    time_start = time.time()
    [CSM, freqs] = developCSM(mic_signal.T, search_freql,
                              search_frequ, framerate, t_start, t_end)
    time_end = time.time()
    print('csm cost', time_end-time_start)

    # % 扫描网格的分辨率   %注:计算转向矢量steerVector和波束成像 -- DAS算法DAS
    scan_resolution = 0.1  # scan_resolution = 0.1;

    time_start = time.time()
    g = steerVector(z_source, freqs, [scan_x, scan_y],
                    scan_resolution, mic_pos.T, c, mic_centre)
    time_end = time.time()
    print('steervector cost', time_end-time_start)

    time_start = time.time()

    # % 波束成像 -- DAS算法
    # [X, Y, B] = DAS(CSM, g, freqs, [scan_x scan_y], scan_resolution);
    [X, Y, B] = DAS(CSM, g, freqs, [scan_x, scan_y], scan_resolution)

    time_end = time.time()
    print('DAS cost', time_end-time_start)
    # % 声压级单位转换
    B[B < 0] = 0
    eps = np.finfo(np.float64).eps
    # SPL = 20*log10((eps+sqrt(real(B)))/2e-5);
    SPL = 20*np.log10((eps+np.sqrt(B.real))/2e-5)
    np.save("testSPL.npy", SPL)
    time_end_total = time.time()
    print('totally cost', time_end_total-time_start_total)

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
        plt.colorbar()  # contourf(X, Y, SPL, (maxSPL-BF_dr):1:maxSPL); colorbar
        # hold on; plot(source_x(:),source_y(:),'r*');
        # plt.plot(source_x.flatten('F'), source_y.flatten('F'), 'r*')
        plt.xlabel('x轴(m)')
        plt.ylabel('y轴(m)')
        plt.title('波束成像图')
        plt.show()

    # plot_figure(X, Y, SPL)
