function g = steerVector(plane_distance, frequencies, scan_limits, grid_resolution, mic_positions, c, mic_centre)
%
% ------ 计算导向矢量
%

% 麦克风个数和扫描频点个数
N_mic = size(mic_positions, 2);
N_freqs = length(frequencies);

% 定义扫描平面
x = scan_limits(1):grid_resolution:scan_limits(2); 
y = scan_limits(3):grid_resolution:scan_limits(4); 
z = plane_distance;
N_X = length(x); N_Y = length(y);
X = repmat(x,N_X,1); Y = repmat(y.',1,N_Y);

% 初始化转向矢量
g = zeros(N_X, N_Y, N_mic, N_freqs);

% 计算扫描平面到麦克风阵列中心的距离
r_scan_to_mic_centre = sqrt((X-mic_centre(1)).^2 + (Y-mic_centre(2)).^2 + (z-mic_centre(3))^2);  

% 初始化变量
r_scan_to_mic = zeros(N_X, N_Y, N_mic);

% 计算转向矢量
for K = 1:N_freqs
    omega = 2*pi*frequencies(K);  % 角频率 w
    for m = 1:N_mic
        r_scan_to_mic(:,:,m) = sqrt((X-mic_positions(1,m)).^2+(Y-mic_positions(2,m)).^2 + z^2); % 计算扫描平面到第 m 个麦克风的距离
        g(:,:,m, K) = (r_scan_to_mic(:,:,m)./r_scan_to_mic_centre).*exp(-1j*omega.*(r_scan_to_mic(:,:,m)-r_scan_to_mic_centre)./c);
    end
end

end