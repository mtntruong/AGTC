import os
import cv2
import torch
import copy
import time
import argparse
import numpy as np
from main_net import RPCA_Net
from util import inv_luma, luma_from_ev, write_EXR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=True,
        help='Path to test data.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to trained weight.',
    )
    parser.add_argument(
        '--output_path',
        default='./HDR_results',
        help='Path to output folder.',
    )
    return parser.parse_args()


def load_pretrained(path, N_iter):
    model = RPCA_Net(N_iter=N_iter)
    model = model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def HDR_inference(model, img0, img1, img2, omg0, omg1, omg2):
    img0 = torch.unsqueeze(torch.from_numpy(img0).cuda(), 0)
    img1 = torch.unsqueeze(torch.from_numpy(img1).cuda(), 0)
    img2 = torch.unsqueeze(torch.from_numpy(img2).cuda(), 0)
    omg0 = torch.unsqueeze(torch.from_numpy(omg0).cuda(), 0)
    omg1 = torch.unsqueeze(torch.from_numpy(omg1).cuda(), 0)
    omg2 = torch.unsqueeze(torch.from_numpy(omg2).cuda(), 0)

    with torch.no_grad():
        x_hat, x_hdr = model(img0, img1, img2, omg0, omg1, omg2)

    hat_patch = torch.squeeze(x_hat).cpu().numpy()
    hdr_patch = torch.squeeze(x_hdr).cpu().numpy()
    return hat_patch, hdr_patch


def create_images(img_folder, out_folder, ckpt_path, N_iter, img_width, img_height):
    # Load model
    model = load_pretrained(ckpt_path, N_iter)

    # Grid
    w_grid = [0]
    h_grid = [0]

    while True:
        if h_grid[-1] + 256 < img_height:
            h_grid.append(h_grid[-1] + 256)
        if w_grid[-1] + 256 < img_width:
            w_grid.append(w_grid[-1] + 256)
        else:
            h_grid[-1] = img_height - 256
            w_grid[-1] = img_width - 256
            break
    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    # HDR reconstruction
    ev = [0.0, 3.0, 6.0]
    gt_img_name = sorted(os.listdir(os.path.join(img_folder, 'HDR')))
    for exr in gt_img_name:
        print(exr)
        seq = exr[0:6]

        # Storing final HDR image
        HDR   = np.float32(np.zeros((img_height, img_width, 3)))
        #HDR_L = np.float32(np.zeros((img_height, img_width, 3)))
        #HDR_M = np.float32(np.zeros((img_height, img_width, 3)))
        #HDR_R = np.float32(np.zeros((img_height, img_width, 3)))

        # Read image stack
        image_short = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'SEQ', seq + '_01.tif'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        image_medium = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'SEQ', seq + '_02.tif'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        image_long = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'SEQ', seq + '_03.tif'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        # Read omega stack
        omega_short = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'OMEGA', seq + '_01.png'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        omega_medium = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'OMEGA', seq + '_02.png'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        omega_long = cv2.cvtColor(cv2.imread(os.path.join(img_folder, 'OMEGA', seq + '_03.png'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        # Patch reconstruction
        i = 0
        j = 0
        total_time = 0
        while i < len(h_grid):
            while j < len(w_grid):
                h = h_grid[i]
                w = w_grid[j]

                image_short_patch = copy.deepcopy(image_short[h:h + 256, w:w + 256, :]) / 65535.0
                image_medium_patch = copy.deepcopy(image_medium[h:h + 256, w:w + 256, :]) / 65535.0
                image_long_patch = copy.deepcopy(image_long[h:h + 256, w:w + 256, :]) / 65535.0
                
                luma_short = luma_from_ev(image_short_patch, ev[0])
                luma_medium = luma_from_ev(image_medium_patch, ev[1])
                luma_long = luma_from_ev(image_long_patch, ev[2])

                img0 = luma_short.astype(np.float32).transpose(2, 0, 1)
                img1 = luma_medium.astype(np.float32).transpose(2, 0, 1)
                img2 = luma_long.astype(np.float32).transpose(2, 0, 1)
                
                omega_short_patch = copy.deepcopy(omega_short[h:h + 256, w:w + 256, :]) / 255.0
                omega_medium_patch = copy.deepcopy(omega_medium[h:h + 256, w:w + 256, :]) / 255.0
                omega_long_patch = copy.deepcopy(omega_long[h:h + 256, w:w + 256, :]) / 255.0

                omg0 = omega_short_patch.astype(np.float32).transpose(2, 0, 1)
                omg1 = omega_medium_patch.astype(np.float32).transpose(2, 0, 1)
                omg2 = omega_long_patch.astype(np.float32).transpose(2, 0, 1)

                # Patch inference and reshape
                start = time.time()
                x_hat, x_hdr = HDR_inference(model, img0, img1, img2, omg0, omg1, omg2)
                total_time += (time.time() - start)

                hat_patch = x_hat.transpose(1, 2, 0)
                hdr_patch = x_hdr.transpose(1, 2, 0)

                # Stitching
                HDR[h:h + 256, w:w + 256, :]   = copy.deepcopy(hdr_patch)

                j = j + 1
            i = i + 1
            j = 0
        
        mask = np.multiply(image_short / 65535.0, image_medium / 65535.0)
        mask = np.amin(mask, axis=2)
        mask[mask < 1] = 0
        mask = np.dstack((mask, mask, mask))
        HDR = inv_luma(HDR)/65535.0
        HDR[mask == 1] = 1.0
        
        # Final writing
        write_EXR(os.path.join(out_folder, seq+'.exr'), HDR)


if __name__ == '__main__':
    
    opt_args = parse_args()
    out_dir = opt_args.output_path
    test_dir = opt_args.data_path
    ckpt_path = opt_args.checkpoint
    
    os.makedirs(out_dir, exist_ok = True)
    create_images(img_folder=test_dir, out_folder=out_dir,
                  ckpt_path=ckpt_path, N_iter=10,
                  img_width=1820, img_height=980)


