import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from util.image_pool import ImagePool
from . import networks
from .residual_cnn_generator import ResidualCNN
import models.shufflenet as ShuffleNet3D
from models.resnet_3d import BasicBlock as BasicBlock3D
from models.resnet_2d import BasicBlock as BasicBlock2D
import torchvision.models
from models.get_model import get_model_recons as get_model

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
            self.loss_names = ['Recons', 'Proj']
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def projection_loss(self, x, y):
        ''' orthogonal projections along each dimension of the generated 3D image
            x: (w, h, d) 3D image
        '''
        x_side0 = torch.mean(x, dim=0)
        x_side1 = torch.mean(x, dim=1)
        x_side2 = torch.mean(x, dim=2)
        y_side0 = torch.mean(y, dim=0)
        y_side1 = torch.mean(y, dim=1)
        y_side2 = torch.mean(y, dim=2)
        loss0 = torch.sum(torch.pow(x_side0 - y_side0, 2))
        loss1 = torch.sum(torch.pow(x_side1 - y_side1, 2))
        loss2 = torch.sum(torch.pow(x_side2 - y_side2, 2))
        return loss0 + loss1 + loss2

    def reconstruction_loss(self, x, y):
        loss = torch.sum(torch.pow(x - y, 2))
        return loss

    def encode_input(self, label_map, real_image=None):
        input_label = torch.tensor(label_map.data.cuda(), device='cuda')
        if real_image is not None:
            real_image = torch.tensor(real_image.data.cuda())
        return input_label, real_image

    def forward(self, input_label, real_image):
        fake_image = self.model.forward(input_label).squeeze()
        real_image = real_image.squeeze()

        loss_Recons = self.reconstruction_loss(fake_image, real_image)
        loss_Proj = self.projection_loss(fake_image, real_image)
        return [[loss_Recons, loss_Proj], fake_image]

    def inference(self, input_label):
        with torch.no_grad():
            fake_image = self.model.forward(input_label)
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

# class InferenceModel(ReconsGANModel):
#     def forward(self, inp):
#         label, inst = inp
#         return self.inference(label, inst)