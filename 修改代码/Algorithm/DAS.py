import numpy as np


def DAS(CSM, g, frequencies, scan_limits, grid_resolution):
    # DAS 算法

    # 参数初始化
    N_freqs = max(frequencies.shape)  # 获取频率个数

    # 扫描平面
    # X = scan_limits(1):grid_resolution:scan_limits(2);
    X = np.arange(scan_limits[0][0], scan_limits[0][1] +
                  grid_resolution, grid_resolution).reshape(1, -1)
    # Y = scan_limits(3):grid_resolution:scan_limits(4);
    Y = np.arange(scan_limits[1][0], scan_limits[1][1] +
                  grid_resolution, grid_resolution).reshape(1, -1)
    N_X = max(X.shape)  # 获取X轴的长度
    N_Y = max(Y.shape)  # 获取Y轴的长度
    # N_X = length(X); N_Y = length(Y); N_mic = size(g,3)
    N_mic = np.size(g, 2)

    # 变量初始化
    B = np.zeros(((N_X, N_Y)), dtype=np.complex128)  # B = zeros(N_X,N_Y);
    # B_freqK = zeros(N_X,N_Y);
    B_freqK = np.zeros((N_X, N_Y), dtype=np.complex128)

    # 计算波束形成的声功率图
    for K in range(1, N_freqs+1):  # for K = 1:N_freqs
        # 频率 K 对应的转向矢量
        gk = g[:, :, :, K-1]  # gk = g(:,:,:,K);

        # 频率 K 下的波束成像图
        for i in range(1, N_X+1):  # for i = 1:N_X
            for j in range(1, N_Y+1):  # for j = 1:N_Y
                gk_xy = np.squeeze(gk[i-1, j-1, :])  # 频率 K 下，位于 xy 位置对应的转向矢量
                # g^{H}Cg
                B_freqK[i-1, j-1] = np.dot(np.dot(gk_xy.conj().T,
                                           CSM[:, :, K-1]), gk_xy)/(N_mic**2)
        # 累加各个频率成分
        B = B + B_freqK
    return [X, Y, B]
