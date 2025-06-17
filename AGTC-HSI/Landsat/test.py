import os
import cv2
import torch
import copy
import time
import argparse
import h5py
import numpy as np
from main_net import RPCA_Net


def load_pretrained(path, N_iter, input_dim):
    model = RPCA_Net(N_iter=N_iter, tensor_num_channels=input_dim)
    model = model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def inference(model, img, omg):
    img0 = torch.unsqueeze(img.cuda(), 0)
    omg0 = torch.unsqueeze(omg.cuda(), 0)

    with torch.no_grad():
        C = model(img0, omg0)

    hsi_patch = torch.squeeze(C).cpu().numpy()
    return hsi_patch


def create_images(ckpt_path, N_iter, input_dim):
    # Load model
    model = load_pretrained(ckpt_path, N_iter, input_dim)

    # Grid
    w_grid = [0, 256]
    h_grid = [0, 256]

    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    f = h5py.File('Landsat_test.mat', 'r')
    reader = f.get('Nmsi')
    data = torch.from_numpy(np.array(reader).astype('float32'))
    reader = f.get('mask')
    omega = torch.from_numpy(np.array(reader).astype('float32'))

    HSI = np.float32(np.zeros((512, 512, 8)))

    # Patch reconstruction
    i = 0
    j = 0
    total_time = 0
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            data_patch = copy.deepcopy(data[:, w:w + 256, h:h + 256])
            omega_patch = copy.deepcopy(omega[:, w:w + 256, h:h + 256])
            
            # Patch inference and reshape
            C = inference(model, data_patch, omega_patch)

            hsi_patch = C.transpose(2, 1, 0)

            # Stitching
            HSI[h:h + 256, w:w + 256, :] = copy.deepcopy(hsi_patch)

            j = j + 1
        i = i + 1
        j = 0
    
    np.save('Landsat-AGTC.npy', HSI)

if __name__ == '__main__':
    
    create_images(ckpt_path='./Weight/AGTC-Landsat.pth', N_iter=10, input_dim=8)

    print('Done')
