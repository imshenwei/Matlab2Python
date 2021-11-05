import math
from matplotlib.pyplot import xcorr
import numpy as np
def  DAS(CSM, g, frequencies, scan_limits, grid_resolution):
    #
    # DAS 算法
    #

    # 参数初始化
    N_freqs = len(frequencies[0])#N_freqs = length(frequencies);

    # 扫描平面
    X = list(range(scan_limits[0],scan_limits[1]+1,grid_resolution))#X = scan_limits(1):grid_resolution:scan_limits(2);
    Y = list(range(scan_limits[2],scan_limits[3]+1,grid_resolution))#Y = scan_limits(3):grid_resolution:scan_limits(4);
    N_X = len(X[0]); N_Y = len(Y[0]); N_mic = size(g,2)#N_X = length(X); N_Y = length(Y); N_mic = size(g,3);

    # 变量初始化
    B = np.zeros((N_X,N_Y))#B = zeros(N_X,N_Y);
    B_freqK = np.zeros((N_X,N_Y))#B_freqK = zeros(N_X,N_Y);

    # 计算波束形成的声功率图
    for K in range(1,N_freqs):#for K = 1:N_freqs  
        # 频率 K 对应的转向矢量
        gk = g[:,:,:,K-1]#gk = g(:,:,:,K);
        
        # 频率 K 下的波束成像图
        for i in range(1,N_X-1)#for i = 1:N_X
            for j in range(1,N_Y-1)#for j = 1:N_Y
                gk_xy = np.squeeze(gk[i-1,j-1,:]); # 频率 K 下，位于 xy 位置对应的转向矢量
                B_freqK(i, j) = np.transpose(gk_xy)*CSM[:,:,K-1]*gk_xy/(N_mic^2);  # g^{H}Cg
        # 累加各个频率成分
        B = B + B_freqK
    return [X, Y, B]