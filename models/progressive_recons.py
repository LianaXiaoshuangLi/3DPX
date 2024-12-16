import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from util.image_pool import ImagePool
from . import networks
import torchvision.models
from models.get_model import get_model_recons as get_model
from monai.transforms import Resize

class ReconsModel(torch.nn.Module):
    def __init__(self, opt):
        super(ReconsModel, self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.model = get_model(opt.model)
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            print('pretrained_path: ', pretrained_path)

        '''loss functions and optimizers'''
        if self.isTrain:
            self.lr = opt.lr
            self.loss_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

    def reconstruction_loss(self, x, y):
        loss = torch.sum(torch.pow(x - y, 2))
        return loss

    def encode_input(self, label_map, real_image=None):
        input_label = torch.tensor(label_map.data.cuda(), device='cuda')
        if real_image is not None:
            real_image = torch.tensor(real_image.data.cuda())
        return input_label, real_image

    def forward(self, input_label, real_image):
        real_image = real_image.squeeze()
        x1, x2, x3, x4, x5, x6, x7, x8 = self.model.forward(input_label)

        real_x1 = Resize((x1.shape[1], x1.shape[2], x1.shape[3]))(real_image)
        real_x2 = Resize((x2.shape[1], x2.shape[2], x2.shape[3]))(real_image)
        real_x3 = Resize((x3.shape[1], x3.shape[2], x3.shape[3]))(real_image)
        real_x4 = Resize((x4.shape[1], x4.shape[2], x4.shape[3]))(real_image)
        real_x5 = Resize((x5.shape[1], x5.shape[2], x5.shape[3]))(real_image)
        real_x6 = Resize((x6.shape[1], x6.shape[2], x6.shape[3]))(real_image)
        real_x7 = Resize((x7.shape[1], x7.shape[2], x7.shape[3]))(real_image)

        loss_x1 = self.reconstruction_loss(x1, real_x1)
        loss_x2 = self.reconstruction_loss(x2, real_x2)
        loss_x3 = self.reconstruction_loss(x3, real_x3)
        loss_x4 = self.reconstruction_loss(x4, real_x4)
        loss_x5 = self.reconstruction_loss(x5, real_x5)
        loss_x6 = self.reconstruction_loss(x6, real_x6)
        loss_x7 = self.reconstruction_loss(x7, real_x7)
        loss_x8 = self.reconstruction_loss(x8, real_image)
        return [[loss_x1, loss_x2, loss_x3, loss_x4, loss_x5, loss_x6, loss_x7, loss_x8], x8]

    def inference(self, input_label):
        with torch.no_grad():
            _,_,_,_,_,_,_,fake_image = self.model.forward(input_label)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.model, 'recons', which_epoch)

    def save_network(self, network, network_label, epoch_label):
        saved_network = [item for item in os.listdir(self.save_dir) if item.endswith('.pth')]
        for item in saved_network:
            if item.startswith(network_label):
                os.remove(os.path.join(self.save_dir, item))
        save_filename = '%s_net_%s.pth' % (network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

class ReconsModel1(torch.nn.Module):
    def __init__(self, opt):
        super(ReconsModel1, self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.model = get_model(opt.model)
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            print('pretrained_path: ', pretrained_path)

        '''loss functions and optimizers'''
        if self.isTrain:
            self.lr = opt.lr

    def forward(self, input_label, real_image):
        real_image = real_image.squeeze()
        loss, fake_image = self.model.forward(input_label, real_image)

        return [loss, fake_image]

    def inference(self, input_label):
        with torch.no_grad():
            fake_image = self.model.inference(input_label)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.model, 'recons', which_epoch)

    def save_network(self, network, network_label, epoch_label):
        saved_network = [item for item in os.listdir(self.save_dir) if item.endswith('.pth')]
        for item in saved_network:
            if item.startswith(network_label):
                os.remove(os.path.join(self.save_dir, item))
        save_filename = '%s_net_%s.pth' % (network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
