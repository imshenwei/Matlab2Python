clc; close all; clear;

% 各种阵列

load('128_limit_7.mat')
figure;plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)

load('128array.mat')
figure; plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)

load('array.mat')
figure; plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)

load('My128.mat')
figure; plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)

load('New64array.mat')
figure; plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)

load('New128array.mat')
figure; plot(array(:,1),array(:,2),'k.', 'MarkerSize',20)
