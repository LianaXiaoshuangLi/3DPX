import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt, isTrain):
    # from data.aligned_dataset import AlignedDataset
    from data.separate_load_dataset import AlignedDataset
    dataset = AlignedDataset()

    dataset.initialize(opt, isTrain)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, isTrain):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, isTrain)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            # shuffle=not opt.serial_batches, # True
            shuffle=False,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
