import torch
import os

def create_model(opt):
    if  opt.model == 'CBCTRecons':
        from .CBCTRecons_model import CBCTReconsModel, InferenceModel
        if opt.isTrain:
            model = CBCTReconsModel()
        else:
            model = InferenceModel()
        model.initialize(opt)
    elif opt.model == 'ResidualGAN':
        from .lsgan_residual import ResidualGANModel, InferenceModel
        if opt.isTrain:
            model = ResidualGANModel()
        else:
            model = InferenceModel()
        model.initialize(opt)
    elif opt.model == 'ResidualCNN':
        from .residual_cnn import ResidualCNN
        model = ResidualCNN()
    elif opt.model == 'ResidualShuffleNetGAN':
        from .residual_shufflenet_gan import ResidualShuffleNetGANModel, InferenceModel
        if opt.isTrain:
            model = ResidualShuffleNetGANModel()
        else:
            model = InferenceModel()
        model.initialize(opt)
    elif opt.model == 'MultiTaskGAN':
        from .multi_task_gan import MultiTaskGANModel, InferenceModel
        if opt.isTrain:
            model = MultiTaskGANModel()
        else:
            model = InferenceModel()
        model.initialize(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.load_pretrain != '':
        state_dict = torch.load(os.path.join('/home/lixiaodian/med/pix2pixHD-cbct/checkpoints', opt.load_pretrain, 'epoch_%s.pth' %opt.which_epoch))
        if opt.isTrain:
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print('load pretrained model from %s' % opt.load_pretrain)
    if opt.continue_train:
        save_path = os.path.join('/home/lixiaodian/med/pix2pixHD-cbct/checkpoints', opt.name)
        saved_network = [item for item in os.listdir(save_path) if item.endswith('.pth')]
        saved_network = sorted(saved_network, key=lambda x: os.path.getmtime(os.path.join(save_path, x)))
        state_dict = torch.load(os.path.join(save_path, saved_network[-1]))
        if opt.isTrain:
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print('continue training from %s' % saved_network[-1])
    return model
