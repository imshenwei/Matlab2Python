import numpy as np
import math


def steerVector(plane_distance, frequencies, scan_limits, grid_resolution, mic_positions, c, mic_centre):
    '''计算转向矢量'''
    # ------ 计算导向矢量
    #

    # %  %注:波束成像 -- DAS算法DAS
    # g = steerVector(z_source, freqs, [scan_x scan_y], scan_resolution, mic_pos.', c, mic_centre);
    # 麦克风个数和扫描频点个数
    N_mic = mic_positions.shape[1]  # N_mic = size(mic_positions, 2);
    N_freqs = max(frequencies.shape)  # N_freqs = length(frequencies)

    # 定义扫描平面  #纯粹range是对象, list应该是对的, 还有+1是因为matlab转换
    # x = scan_limits(1):grid_resolution:scan_limits(2);
    x = np.arange(scan_limits[0][0], scan_limits[0][1] +
                  grid_resolution, grid_resolution).reshape(1, -1)
    # y = scan_limits(3):grid_resolution:scan_limits(4);
    y = np.arange(scan_limits[1][0], scan_limits[1][1] +
                  grid_resolution, grid_resolution).reshape(1, -1)
    z = plane_distance  # z = plane_distance;
    N_X = max(x.shape)
    N_Y = max(y.shape)  # N_X = length(x); N_Y = length(y)
    X = np.kron(np.ones((N_X, 1)), x)
    # X = repmat(x,N_X,1); Y = repmat(y.',1,N_Y)  #.'转置 ,  repmat可能有误
    Y = np.kron(np.ones((1, N_Y)), y.transpose())

    # # 初始化转向矢量
    g = np.zeros((N_X, N_Y, N_mic, N_freqs)).astype(
        np.complex128)  # g = zeros(N_X, N_Y, N_mic, N_freqs);

    # # 计算扫描平面到麦克风阵列中心的距离
    r_scan_to_mic_centre = np.sqrt(np.power(X-mic_centre.flatten('F')[0], 2) + np.power(Y-mic_centre.flatten('F')[1], 2) + np.power(
        z-mic_centre.flatten('F')[2], 2))  # r_scan_to_mic_centre = sqrt((X-mic_centre(1)).^2 + (Y-mic_centre(2)).^2 + (z-mic_centre(3))^2);

    # # 初始化变量
    # r_scan_to_mic = zeros(N_X, N_Y, N_mic);
    r_scan_to_mic = np.zeros((N_X, N_Y, N_mic))

    # 计算转向矢量
    for K in range(1, N_freqs+1):
        omega = 2*math.pi*frequencies.flatten('F')[K-1]  # 角频率 w
        for m in range(1, N_mic+1):
            r_scan_to_mic[:, :, m-1] = np.sqrt(np.power(X-mic_positions[1-1, m-1], 2)+np.power(
                Y-mic_positions[2-1, m-1], 2) + np.power(z, 2))  # 计算扫描平面到第 m 个麦克风的距离
            g[:, :, m-1, K-1] = np.divide(r_scan_to_mic[:, :, m-1], r_scan_to_mic_centre)*np.exp(
                -1j*np.divide(np.dot(omega, (r_scan_to_mic[:, :, m-1]-r_scan_to_mic_centre)), c))
    return g
