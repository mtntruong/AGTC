clear; clc;

load Landsat7_training_clean
load Landsat7_training_mask

gt = double(Landsat7_training_clean) / 139.0;
mask = double(Landsat7_training_mask);
hsi = gt .* mask;

mat_idx = -1;

for idx = 1 : 4500
    pix_x = randi([1 1244]);
    pix_y = randi([1 1144]);
        
    mat_idx = mat_idx + 1;
    mat_name = num2str(mat_idx, '%06d');

    gt_aug = gt(pix_x:pix_x+255, pix_y:pix_y+255, :);
    save(['./Train_Pairs_Landsat/GT/' mat_name '.mat'], 'gt_aug', '-v7.3');

    Nmsi = hsi(pix_x:pix_x+255, pix_y:pix_y+255, :);
    save(['./Train_Pairs_Landsat/HSI/' mat_name '.mat'], 'Nmsi', '-v7.3');
    
    Omega3_3D = mask(pix_x:pix_x+255, pix_y:pix_y+255, :);
    save(['./Train_Pairs_Landsat/OMEGA/' mat_name '.mat'], 'Omega3_3D', '-v7.3');
%     break
end