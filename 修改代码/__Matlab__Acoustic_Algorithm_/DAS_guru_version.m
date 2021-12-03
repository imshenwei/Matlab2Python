function [X, Y, B] = DAS(CSM, g, frequencies, scan_limits, grid_resolution)
%
% DAS 算法
%

% 参数初始化
N_freqs = length(frequencies);

% 扫描平面
X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X); N_Y = length(Y); N_mic = size(g,3);

% 变量初始化
B = zeros(N_X,N_Y);
B_freqK = zeros(N_X,N_Y);

% 计算波束形成的声功率图
for K = 1:N_freqs  
    % 频率 K 对应的转向矢量
    gk = g(:,:,:,K);
    
    % 频率 K 下的波束成像图
    for i = 1:N_X
        for j = 1:N_Y
            gk_xy = squeeze(gk(i,j,:)); % 频率 K 下，位于 xy 位置对应的转向矢量
            B_freqK(i, j) = gk_xy'*CSM(:,:,K)*gk_xy/(N_mic^2);  % g^{H}Cg
        end
    end
    
    % 累加各个频率成分
    B = B + B_freqK;
end


end