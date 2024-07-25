import os
import copy
import shutil
import cv2
import numpy as np

inp_path = './Train/'
out_path = './IMG_Patches/'

img_height = 980
img_width = 1820
w_grid = [0]
h_grid = [0]

while True:
    if h_grid[-1] + 256 < img_height:
        h_grid.append(h_grid[-1] + 128)
    if w_grid[-1] + 256 < img_width:
        w_grid.append(w_grid[-1] + 128)
    else:
        h_grid[-1] = img_height - 256
        w_grid[-1] = img_width - 256
        break
w_grid = np.array(w_grid, dtype=np.uint16)
h_grid = np.array(h_grid, dtype=np.uint16)

count = -1
for idx in range(132):
    
    print(inp_path + "HDR/{:06d}.hdr".format(idx+1))
    
    image_short  = cv2.imread(inp_path + "SEQ/{:06d}_01.tif".format(idx+1), cv2.IMREAD_UNCHANGED)
    image_medium = cv2.imread(inp_path + "SEQ/{:06d}_02.tif".format(idx+1), cv2.IMREAD_UNCHANGED)
    image_long   = cv2.imread(inp_path + "SEQ/{:06d}_03.tif".format(idx+1), cv2.IMREAD_UNCHANGED)
    image_gt     = cv2.cvtColor(cv2.imread(inp_path + "HDR/{:06d}.hdr".format(idx+1), cv2.IMREAD_UNCHANGED),
                                cv2.COLOR_BGR2RGB).astype(np.float32)
    
    omega_short  = cv2.imread(inp_path + "OMEGA_warped/{:06d}_01.png".format(idx+1), cv2.IMREAD_UNCHANGED) / 255.0
    omega_medium = cv2.imread(inp_path + "OMEGA_warped/{:06d}_02.png".format(idx+1), cv2.IMREAD_UNCHANGED) / 255.0
    omega_long   = cv2.imread(inp_path + "OMEGA_warped/{:06d}_03.png".format(idx+1), cv2.IMREAD_UNCHANGED) / 255.0
    
    i = 0
    j = 0
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            image_short_patch  = image_short[h:h + 256, w:w + 256, :]
            image_medium_patch = image_medium[h:h + 256, w:w + 256, :]
            image_long_patch   = image_long[h:h + 256, w:w + 256, :]
            image_gt_patch     = image_gt[h:h + 256, w:w + 256, :]
            
            omega_short_patch  = omega_short[h:h + 256, w:w + 256, :]
            omega_medium_patch = omega_medium[h:h + 256, w:w + 256, :]
            omega_long_patch   = omega_long[h:h + 256, w:w + 256, :]

            count = count + 1
            os.makedirs(os.path.join(out_path, str(count).zfill(6)), exist_ok=True)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '0.tif'), image_short_patch)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '1.tif'), image_medium_patch)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '2.tif'), image_long_patch)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '0.png'), omega_short_patch)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '1.png'), omega_medium_patch)
            cv2.imwrite(os.path.join(out_path, str(count).zfill(6), '2.png'), omega_long_patch)
            np.save(os.path.join(out_path, str(count).zfill(6), 'gt.npy'), image_gt_patch)

            j = j + 1
        i = i + 1
        j = 0
