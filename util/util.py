from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

# Calculate the PSNR for reconstruted 3D volume
def PSNR(pred, gt):
    # mse = np.mean((pred - gt)**2)
    # PIXEL_MAX = 2
    # if mse == 0:
    #     return 100
    # return 10 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # rewrite in torch format
    PIXEL_MAX = torch.tensor(2)
    mse = torch.mean((pred - gt)**2)
    if mse == 0: return torch.tensor(100)
    return 10 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# Calculate the SSIM for reconstruted 3D volume
def SSIM(pred, gt):
    pred = (pred - pred.min()) #/ (pred.max() - pred.min())
    gt = (gt - gt.min()) #/ (gt.max() - gt.min())
    PIXEL_MAX = 2
    C1 = (0.01 * PIXEL_MAX) ** 2
    C2 = (0.03 * PIXEL_MAX) ** 2
    mu1 = np.mean(pred)
    mu2 = np.mean(gt)
    sigma1 = np.std(pred)
    sigma2 = np.std(gt)
    sigma12 = np.mean((pred - mu1) * (gt - mu2))
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
    return ssim

# Calculate the DICE for reconstruted 3D volume
def DICE(pred, gt):
    pred = (pred - torch.min(pred))
    pred = pred / (torch.max(pred) - torch.min(pred))
    gt = gt - torch.min(gt)
    gt = gt / (torch.max(gt) - torch.min(gt))
    avg = torch.mean(gt)
    # turn into 3d volumn by thresholding
    # pred = torch.where(pred > 1.5*avg, torch.ones_like(pred), torch.zeros_like(pred))
    # gt = torch.where(gt > 1.5*avg, torch.ones_like(gt), torch.zeros_like(gt))
    pred = torch.where(pred > avg, torch.ones_like(pred), torch.zeros_like(pred))
    gt = torch.where(gt > avg, torch.ones_like(gt), torch.zeros_like(gt))
    return 2 * torch.sum(pred * gt) / (torch.sum(pred) + torch.sum(gt))

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
