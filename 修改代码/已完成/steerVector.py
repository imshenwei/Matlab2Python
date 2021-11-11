import numpy as np
import math
def steerVector(plane_distance, frequencies, scan_limits, grid_resolution, mic_positions, c, mic_centre):
#
# ------ 计算导向矢量
#

    # 麦克风个数和扫描频点个数
    N_mic = mic_positions.shape[1]#N_mic = size(mic_positions, 2);
    N_freqs = max(frequencies.shape);#N_freqs = length(frequencies);

    # 定义扫描平面  #纯粹range是对象, list应该是对的, 还有+1是因为matlab转换
    x = list(range(scan_limits[0],scan_limits[1]+grid_resolution,grid_resolution))# x = scan_limits(1):grid_resolution:scan_limits(2); 
    y = list(range(scan_limits[2],scan_limits[3]+grid_resolution,grid_resolution)); # y = scan_limits(3):grid_resolution:scan_limits(4); 
    z = plane_distance# z = plane_distance;
    N_X = max(x); N_Y = max(y)# N_X = length(x); N_Y = length(y);
    X = np.tile(x,N_X,1), Y = np.tile(np.transpose(y),1,N_Y)   # X = repmat(x,N_X,1); Y = repmat(y.',1,N_Y)  #.'转置 ,  repmat可能有误

    # # 初始化转向矢量
    g = np.zeros((N_X, N_Y, N_mic, N_freqs))# g = zeros(N_X, N_Y, N_mic, N_freqs);

    # # 计算扫描平面到麦克风阵列中心的距离 
    r_scan_to_mic_centre = math.sqrt(np.power(X-mic_centre[0],2) + np.power(Y-mic_centre[1],2) + np.power(z-mic_centre[2],2))# r_scan_to_mic_centre = sqrt((X-mic_centre(1)).^2 + (Y-mic_centre(2)).^2 + (z-mic_centre(3))^2);  

    # # 初始化变量
    r_scan_to_mic = np.zeros((N_X, N_Y, N_mic))# r_scan_to_mic = zeros(N_X, N_Y, N_mic);

    # 计算转向矢量
    for K in range(1,N_freqs+1):
        omega = 2*math.pi*frequencies[K-1];  # 角频率 w
        for m in range(1,N_mic+1):  #todo:here
            r_scan_to_mic[:,:,m-1] = math.sqrt(np.power(X-mic_positions[1-1,m-1],2)+np.power(Y-mic_positions[2-1,m-1],2) + np.power(z,2)); # 计算扫描平面到第 m 个麦克风的距离
            g[:,:,m-1, K-1] = np.dot(np.devide(r_scan_to_mic[:,:,m-1],r_scan_to_mic_centre),math.exp(-1j*np.divide(np.dot(omega,(r_scan_to_mic[:,:,m-1]-r_scan_to_mic_centre)),c)))
    return g
