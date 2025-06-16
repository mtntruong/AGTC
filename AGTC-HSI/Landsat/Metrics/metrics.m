clc
clear
close all

addpath quality_assess

% Data loading
load Landsat_ground_truth.mat
Landsat_ground_truth = double(Landsat_ground_truth);
Omsi = Landsat_ground_truth ./ max(Landsat_ground_truth(:));

HSI = double(readNPY('Landsat-AGTC.npy'));

[psnr_stripes, ssim_stripes, fsim_stripes, ergas_stripes, msam_stripes] = MSIQA(Omsi*255,HSI*255)
% figure;imagesc(HSI(:,:,[3 5 8]));title('Output')
% figure;imagesc(Omsi(:,:,[3 5 8]));title('GT')
imwrite(uint8( HSI(:,:,[3 5 8])*255), parula(256), 'HSI.png')
% imwrite(uint8(Omsi(:,:,[3 5 8])*255), parula(256), 'gtHSI.png')

error = abs(HSI - Omsi);
error = mean(error, 3);
% figure;imagesc(error);title('Error')
imwrite(uint8(error*7*255), parula(256), 'error.png')
