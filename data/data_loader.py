def CreateDataLoader(opt, isTrain=True):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, isTrain=isTrain)
    return data_loader
