% ----------- DAS 波束成像算法（扫频模式）
% ------ 可以设置多个不同频率、不同声压级的点声源
% ------ 可以设置不同距离的扫描平面
% ------ 可以选择不同的麦克风阵列
% ------ 可以设置想要搜索的频段
% ------ 可以调整网格分辨率

close all; clear; clc;

% 添加路径
addpath('.\Algorithm')
addpath('.\Prepocess')

% 麦克风阵列限定区域
mic_x = [-0.5 0.5];
mic_y = [-0.5 0.5];

% 扫描声源限定区域
scan_x = [-3 3];
scan_y = [-3 3];

% 麦克风阵列平面与扫描屏幕的距离
z_source = 3;

% 声速
c = 343; 

% 信号的采样频率
fs = 8000;

% 导入麦克风阵列
load 56_spiral_array.mat
mic_x_axis = array(:,1); mic_y_axis = array(:,2); mic_z_axis = 0;
mic_pos = [mic_x_axis mic_y_axis]; mic_pos(:,3) = mic_z_axis;
mic_centre = mean(mic_pos); % 阵列中心的坐标

% 绘制麦克风阵列
figure;
plot(mic_x_axis, mic_y_axis,'k.', 'MarkerSize',20);
xlim([min(mic_x_axis)-0.1, max(mic_x_axis)+0.1])
ylim([min(mic_y_axis)-0.1, max(mic_y_axis)+0.1])

% 构建声源点
source_x = [-1,0.5]';
source_y = [0,1]';

% 设定声源频率
source1_freq = 2000;  source2_freq = 3000;
sources_freq = [source1_freq, source2_freq]';

% 设定信号持续时间
t_start = 0;  t_end = 0.02;
source_duration = t_end*ones(length(source_x),1);

% 设定声源声压有效值
source1_spl = 100; source2_spl = 100;  %注意两个不要差太远
sources_spl = [source1_spl, source2_spl].';

% 整合声源信息：source_info
% 声源点坐标x / 声源点坐标y / 声源点坐标z（到扫描平面距离）/ 声源频率/ 声压值
source_info = [source_x, source_y, z_source*ones(length(source_x),1), sources_freq, sources_spl, source_duration]; 

% 获取麦克风阵列输出
mic_signal = simulateArraydata(source_info, mic_pos, c, fs, source_duration, mic_centre);  %

% 确定扫描频段（800-4000 Hz）
search_freql = 800;
search_frequ = 4000;

% 计算CSM以及确定扫描频率
[CSM, freqs] = developCSM(mic_signal.', search_freql, search_frequ, fs, t_start,t_end);

% 扫描网格的分辨率
scan_resolution = 0.1;

% 计算转向矢量
g = steerVector(z_source, freqs, [scan_x scan_y], scan_resolution, mic_pos.', c, mic_centre);

% 波束成像 -- DAS算法
[X, Y, B] = DAS(CSM, g, freqs, [scan_x scan_y], scan_resolution);

% 声压级单位转换
B(B<0)=0;
SPL = 20*log10((eps+sqrt(real(B)))/2e-5);

% 绘制波束成像图
figure;
BF_dr = 6; maxSPL = ceil(max(SPL(:)));
contourf(X, Y, SPL, (maxSPL-BF_dr):1:maxSPL); colorbar;
hold on; plot(source_x(:),source_y(:),'r*'); 
