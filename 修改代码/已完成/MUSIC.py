import math
from matplotlib.pyplot import xcorr
import numpy as np

#matlab2py MUSIC 算法

'''
% Inputs:
%    CSM:  cross-spectrum matrix (CSM)
%    hn:   steering vector
%    nSources:   number of sources
%
% Outputs:
%    MUSIC_result:  beamforming map, obtained by MUSIC
%
% Author: Hao Liang 
% Last modified by: 21/09/15
'''

#setting
def MUSIC(CSM,hn,nSources):
    [N_X,N_Y,N_mic] = hn.shape

    #对角重载
    CSM = CSM + CSM.trace()/(N_mic**2)*np.eye(N_mic)
    CSM = CSM /N_mic

    #特征值
    np.array(Vec,Val) = np.linalg.eigvals(CSM)
    np.array(Seq) = np.sort(max(Val))

    #噪声特征值
    Vn = np.array.Vec(np.array.Seq(range(0,len(np.array.Seq)-1))\
        ,np.array.Seq(range(0,len(np.array.Seq)-nSources-1)))

    #空间频谱成像
    MUSIC_result = np.zeros(N_X,N_Y)
    for ii in range(1,N_X) :
        for jj in range(1,N_Y) :
            e = np.reshape(hn(ii,jj,),N_mic,1)
            MUSIC_result(ii,jj) = 1./(e.T*(Vn*Vn.T)*e)

    return MUSIC_result





