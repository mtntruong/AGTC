% If you encounter the error "Invalid MEX-file"
% Go to hdr-toolbox/source_code/IO, then recompile the .CPP files using MEX
%     mex('read_exr.cpp')
%     mex('write_exr.cpp')

clear; clc;

addpath('./hdrvdp-2.2.2/')
addpath('./zpu21-main/matlab/')
addpath(genpath('./hdr-toolbox/source_code'))

ppd = 30;

gt_dirs = './HDR_GT/'; % This folder contains ground-truth .HDR files
gtNames = dir(fullfile(gt_dirs,'*.hdr'));

hdr_dirs = './HDR_Outputs/'; % This folder contains output .EXR files
imageNames = dir(fullfile(hdr_dirs,'*.exr'));

mu_psnr = zeros(1,length(gtNames));
pu2_psnr = zeros(1,length(gtNames));
pu2_msssim = zeros(1,length(gtNames));
pu2_vsi = zeros(1,length(gtNames));
Q_score = zeros(1,length(gtNames));
P_score = zeros(1,length(gtNames));

parfor i = 1 : length(gtNames)
    inp = hdrread([gt_dirs gtNames(i).name]); inp = inp(51:930,51:1770,:);
    inp = double(inp);
    out = read_exr([hdr_dirs imageNames(i).name]);
    if size(out, 1) == 980
        out = out(51:930,51:1770,:);
    end

    mu_inp = max(0, inp);
    mu_out = max(0, out);
    
    t_inp = log(1 + 5000 * mu_inp) / log(1 + 5000);
    t_out = log(1 + 5000 * mu_out) / log(1 + 5000);
    mu_psnr(i) = psnr( t_out, t_inp );

    % PU21
    mu_inp(mu_inp<=0.0051) = 0.0051;
    mu_out(mu_out<=0.0051) = 0.0051;
    L_peak = 4000;
    pu2_psnr(i) = pu21_metric(mu_inp*L_peak, mu_out*L_peak, 'PSNR');
    pu2_msssim(i) = pu21_metric(mu_inp*L_peak, mu_out*L_peak, 'MSSSIM');
    pu2_vsi(i) = pu21_metric(mu_inp*L_peak, mu_out*L_peak, 'VSI');
    
    % HDR-VDP
    I_context = get_luminance(inp*65535);
    res = hdrvdp( out*65535, inp*65535, 'rgb-bt.709', ppd );
    Q_score(i) = res.Q;
    P_score(i) = mean(res.P_map,'all');
    img = hdrvdp_visualize( res.P_map, I_context );
end

fprintf('mu-psnr: %.4f; pu-psnr: %.4f\n',mean(mu_psnr),mean(pu2_psnr))
fprintf('pu-ssim: %.4f; pu-vsi: %.4f\n',mean(pu2_msssim),mean(pu2_vsi))
fprintf('VDP-P: %.4f; VDP-Q: %.4f\n',mean(Q_score),mean(P_score))
