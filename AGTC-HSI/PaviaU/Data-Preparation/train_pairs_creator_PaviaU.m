clear; clc;

load PaviaU.mat
paviau = paviaU./max(paviaU(:));
Omsi = paviau(1:256,1:256,:); 
rate = 0.5;
Mean = mean(Omsi(:));

M=64;
N=64;
p=103;

mat_idx = -1;

for idx = 1 : 500
    pix_x = randi([257 546]);
    pix_y = randi([257 276]);
    gt_msi = paviau(pix_x:pix_x+63, pix_y:pix_y+63, :);
    
    for aug = 1 : 9
        mat_idx = mat_idx + 1;
        mat_name = num2str(mat_idx, '%06d');
                
        gt_aug = PatchAugmentation(gt_msi, aug);
        save(['./Train_Pairs_PaviaU/GT/' mat_name '.mat'], 'gt_aug', '-v7.3');
                
        [Nmsi,loc]  =  make_stripes(gt_aug, rate, Mean);
        Omega3 = ones(M,N*p);
        Omega3(:,loc) = 0;
        Omega3_3D = reshape(Omega3,[M,N,p]);
        
        completely_gone = randi([1 64], 1, 2);
        Omega3_3D(:,completely_gone,:) = 0;
        
        save(['./Train_Pairs_PaviaU/HSI/' mat_name '.mat'], 'Nmsi', '-v7.3');
        save(['./Train_Pairs_PaviaU/OMEGA/' mat_name '.mat'], 'Omega3_3D', '-v7.3');
    end
%     break
end