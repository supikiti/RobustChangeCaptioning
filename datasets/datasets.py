from torch.utils.data import DataLoader

def create_dataset(cfg, split='train', img_feat_base_path=None):
    dataset = None
    data_loader = None
    if cfg.data.dataset == 'rcc_dataset':
        from datasets.rcc_dataset import RCCDataset, RCCDataLoader

        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)

    elif cfg.data.dataset == "hag_dataset":
        from datasets.hag_dataset import HAGDataset

        dataset = HAGDataset(cfg, split, img_feat_base_path)
        data_loader = DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == "hag_dataset_with_scene":
        from datasets.hag_dataset_with_scene import HAGDataset_with_scene

        dataset = HAGDataset_with_scene(cfg, split, img_feat_base_path)
        data_loader = DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == "hag_dataset_with_rcnn":
        from datasets.hag_dataset_with_rcnn import HAGDataset_with_rcnn

        dataset = HAGDataset_with_rcnn(cfg, split, img_feat_base_path)
        data_loader = DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    else:
        raise Exception('Unknown dataset: %s' % cfg.data.dataset)
    
    return dataset, data_loader
