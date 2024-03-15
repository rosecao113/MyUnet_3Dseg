import os
from glob import glob

import numpy as np
import torch
import pandas as pd
from ignite.engine import _prepare_batch
from monai.data import Dataset, pad_list_data_collate, decollate_batch, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    Spacingd,
    RandShiftIntensityd,
    RandRotated,
    ScaleIntensityd,
)


def read_nifti_data(data_dir):
    # read nifti files and create train/val/test list w.r.t split
    # split_name = os.path.join(data_dir, 'split.csv')
    # nifti_data_array = np.array(pd.read_csv(split_name, header=None))
    # names = nifti_data_array[:, 0].tolist()
    # tags = nifti_data_array[:, 1].tolist()
    # train_ds = [{'img': os.path.join(data_dir, fn), 'seg': os.path.join(data_dir, fn.replace('image', 'label'))}
    #             for fn, tag in zip(names, tags) if tag == 'train']
    # val_ds = [{'img': os.path.join(data_dir, fn), 'seg': os.path.join(data_dir, fn.replace('image', 'label'))}
    #           for fn, tag in zip(names, tags) if tag == 'val']
    # test_ds = [{'img': os.path.join(data_dir, fn), 'seg': os.path.join(data_dir, fn.replace('image', 'label'))}
    #            for fn, tag in zip(names, tags) if tag == 'test']
    # return (sorted(train_ds, key=lambda x: x['img']), sorted(val_ds, key=lambda x: x['img']),
    #         sorted(test_ds, key=lambda x: x['img']))

    # read nifti files for either training or validation
    # return single dataset dict
    image_list = sorted(glob(os.path.join(data_dir, "image*.nii.gz")))
    label_list = sorted(glob(os.path.join(data_dir, "labels*.nii.gz")))
    dataset = [{'img': image, 'seg': label} for image, label in zip(image_list, label_list)]
    return dataset


def setup_data(data_dir, train=True):
    files = read_nifti_data(data_dir)

    # data augmentation for training
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=['img', 'seg']),
        # nibabel's LPS = ITK's RAI
        Orientationd(keys=["img", "seg"], axcodes="LPS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["img"]),
        SpatialPadd(keys=["img", "seg"], spatial_size=(64, 64, 64)),
        # the order of axes is inverse to sitk's, (n,c,x,y,z) here
        RandRotated(keys=["img", "seg"], range_x=.5, range_y=.2, range_z=.2, mode=("bilinear", "nearest")),
        # num_samples = batch_size, increasing efficiency
        RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=(64,64,64), pos=1,
                               neg=1, num_samples=2, image_key="img", image_threshold=0),
        RandShiftIntensityd(keys=["img"], offsets=0.10, prob=0.50),
    ])

    # data augmentation for validation/testing
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="LPS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["img"]),
        SpatialPadd(keys=["img", "seg"], spatial_size=(128, 128, 128)),
    ])

    if train:
        train_ds = Dataset(data=files, transform=train_transforms)
        loader = DataLoader(
            train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        val_ds = Dataset(data=files, transform=val_transforms)
        loader = DataLoader(
            val_ds,
            shuffle=False,
            batch_size=5,
            num_workers=4,
            collate_fn=pad_list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

    # check dataset
    # print(f"\n number of subjects: {len(files)}.\n"
    #        f"The first element in the dataset is {files[0]}.")
    # dict_loader = LoadImaged(keys=["img","seg"])
    # for i in range(len(train_files)):
    #      data_dict = dict_loader(train_files[i])
    #      print(f"image shape: {data_dict['img'].shape}, \t label shape: {data_dict['seg'].shape}")

    # test_ds = Dataset(data=test_files, transform=val_transforms)
    # test_loader = DataLoader(
    #     test_ds,
    #     shuffle=False,
    #     batch_size=5,
    #     num_workers=4,
    #     collate_fn=pad_list_data_collate,
    #     pin_memory=torch.cuda.is_available(),
    # )
    return loader


# batch=(img, seg), returns output=loss
def prepare_batch(batch, device=None, non_blocking=False):
    return _prepare_batch((batch["img"], batch["seg"]), device, non_blocking)
