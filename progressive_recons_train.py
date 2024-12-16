import time
import os
import numpy as np
import torch
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
from options.cbct_train_options import TrainOptions
import util.util as util
import util.ssim3d as ssim3d
from util.visualizer import Visualizer
from data.separate_load_dataset import AlignedDataset
import warnings
warnings.filterwarnings("ignore")
import monai
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize, Activations, AsDiscrete
from monai.data import ImageDataset, DataLoader, decollate_batch
from glob import glob
import nibabel as nib
import nrrd

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
opt = TrainOptions().parse()
log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
os.system('cp recons_train.py %s' % os.path.join(opt.checkpoints_dir, opt.name))
os.system('cp models/%s.py %s' % (opt.model, os.path.join(opt.checkpoints_dir, opt.name)))

start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

print('---------- Loadind Dataset  -------------')
start_t = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')
trans_3D = Compose([ScaleIntensity(minv=0, maxv=1),EnsureChannelFirst(),Resize((128, 128, 256)),])
trans_2D = Compose([ScaleIntensity(minv=0, maxv=1),EnsureChannelFirst(),])
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

images_3D = sorted(glob(os.path.join('datasets/recons/train', "*_3D.nii.gz")))
images_2D = sorted(glob(os.path.join('datasets/recons/train', "*_2D.nii.gz")))
train_dataset = ImageDataset(images_3D, images_2D, transform=trans_3D, seg_transform=trans_2D)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=4, pin_memory=False, shuffle=True)
print('train dataset size: %d' % len(train_dataset))

images_3D = sorted(glob(os.path.join('datasets/recons/val', "*_3D.nii.gz")))
images_2D = sorted(glob(os.path.join('datasets/recons/val', "*_2D.nii.gz")))
val_dataset = ImageDataset(images_3D, images_2D, transform=trans_3D, seg_transform=trans_2D)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batchSize, num_workers=4, pin_memory=False, shuffle=False)
print('val dataset size: %d' % len(val_dataset))
print('---------- Loadind Dataset Done, Time: %.2f -------------' % (time.time() - start_t))

model = __import__('models.progressive_recons', fromlist=[''])
model = model.ReconsModel1(opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

visualizer = Visualizer(opt)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)


message = ', '.join([f'{k}={v}' for k, v in vars(opt).items()])
print(message)
with open(log_name, "a") as log_file: log_file.write(message + '\n')

total_steps = (start_epoch-1) * len(train_dataset) + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

max_recons_overall = 0
save_path = os.path.join(opt.checkpoints_dir, opt.name)
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    ''' Train '''
    model.train()
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % len(train_dataset)
    for i, (data_3D, data_2D) in enumerate(train_dataloader):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        input = [torch.tensor(data_2D, dtype=torch.float32).cuda(), torch.tensor(data_3D, dtype=torch.float32).cuda()]
        loss, generated = model(*input)

        loss_dict = dict({'loss': loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if total_steps % opt.print_freq == 0:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) #/ opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        if epoch_iter >= len(train_dataset): break

    if epoch >= 0:
        with torch.no_grad():
            ''' EVAL'''
            psnr = []
            dice = []
            ssim = []
            overall = []
            model.eval()
            for i, data in enumerate(val_dataloader):
                data_3D, data_2D = data
                data_3D = torch.tensor(data_3D, dtype=torch.float32).cuda().squeeze(1)
                data_2D = torch.tensor(data_2D, dtype=torch.float32).cuda()

                generated = model.module.inference(data_2D)
                psnr.append(util.PSNR(generated, data_3D))
                dice.append(util.DICE(generated, data_3D))
                ssim.append(ssim3d.ssim3D(generated, data_3D))
                overall.append((psnr[-1]/20 + dice[-1] + ssim[-1])/3)

            psnr = torch.mean(torch.stack(psnr)).detach().cpu().numpy()
            dice = torch.mean(torch.stack(dice)).detach().cpu().numpy()
            ssim = torch.mean(torch.stack(ssim)).detach().cpu().numpy()
            overall = torch.mean(torch.stack(overall)).detach().cpu().numpy()
            message = 'lr:%.6f PSNR: %.3f, DICE: %.3f, SSIM: %.3f, overall: %.3f' \
                      % (optimizer.param_groups[0]['lr'], np.mean(psnr), np.mean(dice) * 100, np.mean(ssim) * 100, overall * 100)
            print(message)
            with open(log_name, "a") as log_file: log_file.write(message + '\n')

            ''' save '''
            if max_recons_overall < overall:
                train_cls_epoch = 6
                message = 'saving recons model at the end of epoch %d, iters %d' % (epoch, total_steps)
                print(message)
                with open(log_name, "a") as log_file: log_file.write(message + '\n')
                model.module.save(epoch)
                max_recons_overall = overall
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
