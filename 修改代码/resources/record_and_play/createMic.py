'''绘制麦克风阵列

提供了以下功能:
1. 麦克风自定义设置位置
2. 绘制麦克风位置
'''
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

array = np.array(np.zeros((6, 2)))

r = 4.6
array[0, 0] = +0.0460
array[0, 1] = +0.0000

array[1, 0] = +0.0230
array[1, 1] = -0.0398

array[2, 0] = -0.0230
array[2, 1] = -0.0398

array[3, 0] = -0.0460
array[3, 1] = +0.0000

array[4, 0] = -0.0230
array[4, 1] = +0.0398

array[5, 0] = +0.0230
array[5, 1] = +0.0398

mat_path = '6_spiral_array.mat'

io.savemat(mat_path, {'array': array})

#  绘制麦克风阵列
mic_x_axis = array[:, 0]
mic_y_axis = array[:, 1]
mic_z_axis = 0
mic_pos = np.transpose([mic_x_axis, mic_y_axis])
mic_pos = np.concatenate(
    (mic_pos, np.ones((mic_x_axis.size, 1))*mic_z_axis), axis=1)
# 阵列中心的坐标
mic_centre = mic_pos.mean(axis=0).reshape(1, -1)
plt.plot(mic_x_axis, mic_y_axis, 'k.', markersize=20)
plt.xlim([min(mic_x_axis)-0.1, max(mic_x_axis)+0.1])
plt.ylim([min(mic_y_axis)-0.1, max(mic_y_axis)+0.1])
plt.show()
