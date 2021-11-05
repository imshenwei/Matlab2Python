import numpy as np
from numpy.linalg import norm
import math
def simulateArraydata(source_info, mic_info, c, fs, duration, mic_centre):
    #
    # 生成麦克风阵列采集的时域数据
    #

    # 声源和麦克风传感器的个数
    N_source = size(source_info, 0)#N_source = size(source_info, 1); 
    N_mic = size(mic_info, 0)#N_mic = size(mic_info, 1); 

    # 计算样本点个数（分帧）
    t = list(range(0,(duration-1/fs)+1,1/fs))#t = 0:1/fs:(duration-1/fs);
    N_samples = len(t[0])#N_samples = length(t);

    # 麦克风阵列采集数据的初始化
    mic_signal = np.zeros((N_mic,N_samples))#mic_signal = zeros(N_mic, N_samples)
    # 对每个声源累加声音信号（不相干声源假设）
    for I in range(1,N_source+1):
        # 阵列中心到声源的距离
        r_source_to_centre = norm(mic_centre-source_info[I-1, 0:2])
        # 信号功率为 amp^2
        # 声压级 SPL = 20*log10(amp/2e-5), 我们通过从源到阵列中心的距离进行缩放以获得中心处的给定 SPL
        amp = 2e-5*10^(source_info[I-1, 4]/20)
        for J in range(1,N_mic+1):
            # 第 J 个麦克风到仿真声源点的距离
            r_source_to_mic = math.sqrt(np.dot(mic_info[J-1, :] - source_info[I-1, 0:2], mic_info[J-1, :] - source_info[I-1, 0:2]) )
            # 生成第 J 个麦克风采集的信号
            delay_time = (r_source_to_mic-r_source_to_centre)/c; # 延迟时间
            mic_signal[J-1, :] = mic_signal[J-1, :] + math.sqrt(2)*amp*cos(2*math.pi*source_info[I-1, 3]*(t-delay_time))*r_source_to_centre/r_source_to_mic  
    return mic_signal