import time
import os
import numpy as np
import torch
import math
from options.test_options import TestOptions
import util.util as util
import util.ssim3d as ssim3d
from util.visualizer import Visualizer
import warnings
warnings.filterwarnings("ignore")
import monai
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize, Activations, AsDiscrete
from monai.data import ImageDataset, DataLoader, decollate_batch
from glob import glob

opt = TestOptions().parse(save=False)
opt_path = os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'opt.txt')
for line in open(opt_path, 'r').readlines():
    line = line.strip().split(':')
    if line[0] == 'model': opt.model = line[1].strip()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_name = os.path.join(opt.checkpoints_dir, '%s_test_log.txt' %opt.load_pretrain)

print('---------- Loadind Dataset  -------------')
start_t = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')
trans_3D = Compose([ScaleIntensity(minv=0, maxv=1),EnsureChannelFirst(),Resize((128, 128, 256)),])
trans_2D = Compose([ScaleIntensity(minv=0, maxv=1),EnsureChannelFirst(),])
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

images_3D = sorted(glob(os.path.join('datasets/recons/test', "*_3D.nii.gz")))
images_2D = sorted(glob(os.path.join('datasets/recons/test', "*_2D.nii.gz")))
test_dataset = ImageDataset(images_3D, images_2D, transform=trans_3D, seg_transform=trans_2D)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, num_workers=4, pin_memory=False, shuffle=False)
print('test dataset size: %d' % len(test_dataset))
print('---------- Loadind Dataset Done, Time: %.2f -------------' % (time.time() - start_t))

model = __import__('models.progressive_recons', fromlist=[''])
model = model.ReconsModel1(opt).cuda()

load_path = os.path.join(opt.checkpoints_dir, opt.load_pretrain)

path = [os.path.join(load_path, x) for x in os.listdir(load_path) if x.endswith('.pth')][0]
model.model.load_state_dict(torch.load(path))
print('load pretrained model from %s' % opt.load_pretrain)

model.eval()
overall_list = []

with torch.no_grad():
    ''' TEST '''
    psnr = []
    dice = []
    ssim = []
    overall = []
    model.eval()
    for i, data in enumerate(test_dataloader):
        data_3D, data_2D = data
        data_3D = torch.tensor(data_3D, dtype=torch.float32).cuda().squeeze(0)
        data_2D = torch.tensor(data_2D, dtype=torch.float32).cuda()

        generated = model.inference(data_2D)
        if len(generated.shape) == 3: generated = generated.unsqueeze(0)
        psnr.append(util.PSNR(generated, data_3D))
        dice.append(util.DICE(generated, data_3D))
        ssim.append(ssim3d.ssim3D(generated, data_3D))
        overall.append((psnr[-1]/20 + dice[-1] + ssim[-1])/3)

    psnr = torch.stack(psnr).detach().cpu().numpy()
    dice = torch.stack(dice).detach().cpu().numpy()
    ssim = torch.stack(ssim).detach().cpu().numpy()
    overall = torch.mean(torch.stack(overall)).detach().cpu().numpy()
    message = 'TEST: PSNR: %.3f, DICE: %.3f, SSIM: %.3f, overall: %.3f' \
              % (np.mean(psnr), np.mean(dice) * 100, np.mean(ssim) * 100, np.mean(overall) * 100)
    print(message)
    with open(log_name, "a") as log_file: log_file.write(message + '\n')

    test_overall = overall * 100
    psnr_mean, psnr_std = np.mean(psnr), np.std(psnr)
    dice_mean, dice_std = np.mean(dice)*100, np.std(dice)*100
    ssim_mean, ssim_std = np.mean(ssim)*100, np.std(ssim)*100
    message = 'psnr, dice, ssim: %.2f, %.2f, %.2f' % (psnr_mean, dice_mean, ssim_mean)
    print(message)
    with open(log_name, "a") as log_file: log_file.write(message + '\n')