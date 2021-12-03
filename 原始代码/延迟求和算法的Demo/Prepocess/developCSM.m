function [CSM, freqs] = developCSM(mic_signal, search_freql, search_frequ, Fs, t_start, t_end)
%
% 生成互谱矩阵 CSM
%

% 麦克风阵列数
N_mic = size(mic_signal, 2);

% 开始和结束的样本点
start_sample = floor(t_start*Fs) + 1;
end_samples = ceil(t_end*Fs);  

% 选取在扫描频率之间的点
x_fr = Fs / end_samples * (0:floor(end_samples/2)-1);
freq_sels = find((x_fr>=search_freql).*(x_fr<=search_frequ));

% 扫描频点的个数
N_freqs = length(freq_sels);

% 初始化互谱矩阵 CSM
CSM = zeros(N_mic, N_mic, N_freqs);

% 对采集到的时域数据进行傅里叶变换
mic_signal_fft = sqrt(2)*fft(mic_signal(start_sample:end_samples,:))/(end_samples-start_sample+1);

% 生成互谱矩阵 CSM
for K = 1:N_freqs
    % 计算第 K 个频率下的互谱矩阵
    CSM(:,:,K) = mic_signal_fft(freq_sels(K),:).'*conj(mic_signal_fft(freq_sels(K),:));
%     CSM(:,:,K) = CSM(:,:,K) - diag(diag(CSM(:,:,K)));  % 对角线擦除技术
end
    
% 扫描的频点
freqs = x_fr(freq_sels);

end