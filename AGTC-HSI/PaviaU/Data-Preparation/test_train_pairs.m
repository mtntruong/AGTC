clear; clc; close;

load 'Train_Pairs_PaviaU/GT/000007.mat';
load 'Train_Pairs_PaviaU/HSI/000007.mat';
load 'Train_Pairs_PaviaU/OMEGA/000007.mat';

idx = 50;

imshow(gt_aug(:,:,idx))
figure
imshow(Nmsi(:,:,idx))
figure
imshow(Omega3_3D(:,:,idx))