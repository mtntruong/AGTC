import os
import cv2
import copy
import OpenEXR
import Imath
import numpy as np
import torch
import glob
import h5py
from torch.utils.data import Dataset


def read_EXR(filename):
    exr = OpenEXR.InputFile(filename)
    header = exr.header()
    dw = header['dataWindow']
    img_size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channelData = dict()
    for c in header['channels']:
        C = exr.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, img_size)
        channelData[c] = C
    exr.close()
    img = np.zeros((dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1, 3))
    img[:,:,0] = channelData['R']
    img[:,:,1] = channelData['G']
    img[:,:,2] = channelData['B']
    return img


def write_EXR(filename, hdr_data):
    height, width, _ = hdr_data.shape
    channel_names = ['R', 'G', 'B']
    header = OpenEXR.Header(width, height)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    header['channels'] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channel_names}
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({c: hdr_data[:, :, i].astype(np.float32).tostring() for i, c in enumerate(channel_names)})
    out.close()


def luma(input):
    a = 17.554; b = 826.81; c = 0.10013; d = -884.17; e = 209.16; f = -731.28
    yl = 5.6046; yh = 10469
    output = copy.deepcopy(input)
    output[input < yl] = a * input[input < yl]
    output[(input >= yl) & (input < yh)] = b * np.power(input[(input >= yl) & (input < yh)], c) + d
    output[input >= yh] = e * np.log(input[input >= yh]) + f
    output = output / 4096.0
    return output


def inv_luma(input):
    a = 0.056968; b = 7.3014e-30; c = 9.9872; d = 884.17; e = 32.994; f = 0.0047811
    ll = 98.381; lh = 1204.7
    new_input = copy.deepcopy(input)
    new_input = new_input * 4096.0
    output = copy.deepcopy(new_input)
    output[new_input < ll] = a * new_input[new_input < ll]
    output[(new_input >= ll) & (new_input < lh)] = b * np.power((new_input[(new_input >= ll) & (new_input < lh)] + d), c)
    output[new_input >= lh] = e * np.exp(f * new_input[new_input >= lh])
    return output


def luma_from_ev(inp_channel, ev):
    channel = copy.deepcopy(inp_channel)
    rad_channel = (channel ** 2.2) / (2 ** ev)
    rad_channel = luma(rad_channel * 65535.0)
    return rad_channel


def rad_from_ev(inp_channel, ev):
    channel = copy.deepcopy(inp_channel)
    rad_channel = (channel ** 2.2) / (2 ** ev)
    return rad_channel


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.cvtColor(cv2.imread(imgStr, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 65535.0
        imgs.append(img)
    return np.array(imgs)


def ReadOmegas(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.cvtColor(cv2.imread(imgStr, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return np.array(imgs)


class MainNetDataset(Dataset):
    def __init__(self, root_dir):
        self.exposures = np.array([0, 3, 6])
        self.root_dir = root_dir
        self.scenes_dir_list = sorted(os.listdir(self.root_dir))
        self.image_list = []
        for scene in range(len(self.scenes_dir_list)):
            image_path = sorted(glob.glob(os.path.join(self.root_dir, self.scenes_dir_list[scene], '*.tif')))
            omega_path = sorted(glob.glob(os.path.join(self.root_dir, self.scenes_dir_list[scene], '*.png')))
            hdr_path   = os.path.join(self.root_dir, self.scenes_dir_list[scene], 'gt.npy')
            self.image_list += [[image_path, omega_path, hdr_path]]

    def __getitem__(self, index):
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][0])
        omg_images = ReadOmegas(self.image_list[index][1])

        # Read HDR label
        label = np.load(self.image_list[index][2])

        # Process LDR images
        image_short = luma_from_ev(ldr_images[0], self.exposures[0])
        image_medium = luma_from_ev(ldr_images[1], self.exposures[1])
        image_long = luma_from_ev(ldr_images[2], self.exposures[2])
        img0 = image_short.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long.astype(np.float32).transpose(2, 0, 1)

        # Process Omegas
        omg0 = omg_images[0].astype(np.float32).transpose(2, 0, 1)
        omg1 = omg_images[1].astype(np.float32).transpose(2, 0, 1)
        omg2 = omg_images[2].astype(np.float32).transpose(2, 0, 1)

        # Process HDR label
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = luma(label * 65535.0)

        # To torch tensor
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        omg0 = torch.from_numpy(omg0)
        omg1 = torch.from_numpy(omg1)
        omg2 = torch.from_numpy(omg2)
        label = torch.from_numpy(label)

        return img0, img1, img2, omg0, omg1, omg2, label

    def __len__(self):
        return len(os.listdir(self.root_dir))


class HSIDataset(Dataset):
    def __init__(self, path):
        self.gt_path = os.path.join(path, 'GT')
        self.omega_path = os.path.join(path, 'OMEGA')
        self.seq_path = os.path.join(path, 'HSI')
        self.mat_list = os.listdir(self.gt_path)

    def __getitem__(self, index):
        f = h5py.File(os.path.join(self.seq_path, self.mat_list[index]), 'r')
        reader = f.get('Nmsi')
        data = torch.from_numpy(np.array(reader).astype('float32'))

        f = h5py.File(os.path.join(self.omega_path, self.mat_list[index]), 'r')
        reader = f.get('Omega3_3D')
        omega = torch.from_numpy(np.array(reader).astype('float32'))

        f = h5py.File(os.path.join(self.gt_path, self.mat_list[index]), 'r')
        reader = f.get('gt_aug')
        target = torch.from_numpy(np.array(reader).astype('float32'))

        return data, omega, target

    def __len__(self):
        return len([1 for x in list(os.scandir(self.gt_path)) if x.is_file()])
