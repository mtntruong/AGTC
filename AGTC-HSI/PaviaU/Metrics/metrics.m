clc
clear
close all

addpath quality_assess

% Data loading
load PaviaU_ground_truth.mat
paviau = paviaU ./ max(paviaU(:));
Omsi   = paviau(1:256,1:256,:); 

HSI = double(readNPY('Pavia-AGTC.npy'));

[psnr_stripes, ssim_stripes, fsim_stripes, ergas_stripes, msam_stripes] = MSIQA(Omsi*255,HSI*255)
% figure;imagesc(HSI(:,:,[33 75 68]));title('Output')
% figure;imagesc(Omsi(:,:,[33 75 68]));title('GT')
imwrite(uint8( HSI(:,:,[33 75 68])*255), parula(256), 'HSI.png')
% imwrite(uint8(Omsi(:,:,[33 75 68])*255), parula(256), 'gtHSI.png')

error = abs(HSI - Omsi);
error = mean(error, 3);
% figure;imagesc(error);title('Error')
imwrite(uint8(error*7*255), parula(256), 'error.png')
