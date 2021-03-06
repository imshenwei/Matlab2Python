% ------------------------ A Demo for Beamforming -------------------------
% 
% A simple demo for acoustic imaging, including the following methods:
%
% -- DAS
% -- MUSIC
% -- DAMAS
% -- CLEAN-PSF
% -- CLEAN-SC
% -- FFT-NNLS 
%
% Reference: 
% -- https://github.com/jorgengrythe/beamforming
% -- https://github.com/Anwar-M/Acoustic-Beamforming
%
% Author: Hao Liang 
% Last modified by: 21/10/27
%

%% Experiment setup
clc; clear; close all;
load('56_spiral_array.mat');   % load microphone array
rn = array;  % spatial location of microphones
N = 50;      % number of grid points in each dim
z0 = 15;      % source distance 
phi = 15;    % off-axis angle 
f = 2500;    % imaging frequency 
SNR = 15;    % signal-to-noise ratio (SNR)
% source = int64([N/4 N/4]);    % x,y position of sources
 source = int64([N/4 N/4;N/4 N*3/4]);  % Two sources example

%% DAS
[DAS_result, PSF, hn, CSM] = DAS(N,z0,f,phi,rn,source,SNR);

figure; contourf(real(DAS_result)); title('DAS')
hold on; plot(source(:,1),source(:,2),'r*');

%% MUSIC
nSources = 2;
MUSIC_result = MUSIC(CSM, hn, nSources);

figure; contourf(real(MUSIC_result)); title('MUSIC')
hold on; plot(source(:,1),source(:,2),'r*');

%% DAMAS
maxIter = 100;
DAMAS_result = DAMAS(DAS_result, hn, maxIter);

figure; contourf(real(DAMAS_result)); title('DAMAS')
hold on; plot(source(:,1),source(:,2),'r*')

%% CLEAN-PSF
loopgain = 0.9; maxIter = 100;
CLEAN_PSF_result = CLEAN_PSF(loopgain, maxIter, CSM, hn);

figure; contourf(real(CLEAN_PSF_result)); title('CLEAN-PSF')
hold on; plot(source(:,1),source(:,2),'r*')

%% CLEAN-SC
loopgain = 0.9; maxIter = 100;
CLEAN_SC_result = CLEAN_SC(loopgain, maxIter, CSM, hn);

figure; contourf(real(CLEAN_SC_result)); title('CLEAN-SC')
hold on; plot(source(:,1),source(:,2),'r*')

%% FFT-NNLS
maxIter = 100; 
FFT_NNLS_result = FFT_NNLS(DAS_result, PSF, maxIter);

figure; contourf(real(FFT_NNLS_result)); title('FFT-NNLS')
hold on; plot(source(:,1),source(:,2),'r*')


