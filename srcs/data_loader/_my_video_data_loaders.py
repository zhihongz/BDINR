import sys
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from os.path import isfile as opif
# =================
# loading multiple frames from a video
# =================

# =================
# basic functions
# =================


def vid_transform(vid, prob=0.5, tform_op=['all']):
    """
    video data vid_transform (data augment) with a $op chance

    Args:
        vid ([ndarray]): [shape: N*H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            vid = vid[:, :, ::-1, :]
        if np.random.rand() < prob:
            vid = vid[:, ::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(vid, axes=(0, 2, 1, 3))[:, ::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            vid = np.transpose(
                vid[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]  # 90

    if 'reverse' in tform_op or 'all' in tform_op:
        if np.random.rand() < prob:
            vid = vid[::-1, ...]
    
    return vid.copy()

# =================
# Video Dataset
# =================


class VideoFrame_Dataset(Dataset):
    """
    datasetfor training or test (with ground truth)
    """

    def __init__(self, data_dir, frame_num, patch_sz=None, tform_op=None, sigma_range=0, stride=1):
        super(VideoFrame_Dataset, self).__init__()
        self.sigma_range = sigma_range
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.vid_length = frame_num
        self.img_paths = []
        self.vid_idx = []
        self.stride = stride  # stride of the starting frame

        # get image paths
        img_nums = []
        vid_paths = []
        if isinstance(data_dir, str):
            # single dataset
            vid_names = sorted(os.listdir(data_dir))
            vid_paths = [opj(data_dir, vid_name) for vid_name in vid_names]
            if all(opif(vid_path) for vid_path in vid_paths):
                # data_dir is an image dir rather than a vid dir
                vid_paths = [data_dir]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                vid_names_n = sorted(os.listdir(data_dir_n))
                vid_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in vid_names_n]
                vid_paths.extend(vid_paths_n)

        for vid_path in vid_paths:
            img_names = sorted(os.listdir(vid_path))
            img_nums.append(len(img_names))
            self.img_paths.extend(
                [opj(vid_path, img_name) for img_name in img_names])

        counter = 0
        for img_num in img_nums:
            self.vid_idx.extend(
                list(range(counter, counter+img_num-self.vid_length+1, stride)))
            counter = counter+img_num

    def __getitem__(self, idx):
        # load video frames
        vid = []
        for k in range(self.vid_idx[idx], self.vid_idx[idx]+self.vid_length):
            # read image
            img = cv2.imread(self.img_paths[k])
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.patch_sz:
                if k == self.vid_idx[idx]:
                    # set the random crop point
                    img_sz = img.shape
                    assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                                ), 'error PATCH_SZ larger than image size'
                    xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
                    ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])

                # crop to patch size
                img_crop = img[ymin:ymin+self.patch_sz[0],
                               xmin:xmin+self.patch_sz[1], :]
            else:
                img_crop = img

            vid.append(img_crop)

        # list2ndarray, shape [vid_num, h, w, c], value 0-255
        vid = np.array(vid)  

        # data augment
        if self.tform_op:
            vid = vid_transform(vid, tform_op=self.tform_op)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level>0:
            image_dtype = vid.dtype
            image_maxv = np.iinfo(image_dtype).max  # 8/16 bit image -> 255/65535
            vid = vid + np.random.normal(0, image_maxv*noise_level, vid.shape)
            vid = vid.clip(0, image_maxv).astype(image_dtype)

        return vid.transpose(0, 3, 1, 2)

    def __len__(self):
        return len(self.vid_idx)


class VideoFrame_Dataset_all2CPU(Dataset):
    """
    Dataset for training or test (with ground truth), load entire dataset to CPU to speed the data load process
    """

    def __init__(self, data_dir, frame_num, patch_sz=None, tform_op=None, sigma_range=0, stride=1):
        super(VideoFrame_Dataset_all2CPU, self).__init__()
        self.sigma_range = sigma_range
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.vid_length = frame_num
        self.img_paths = []
        self.vid_idx = []  # start frame index of each video
        self.imgs = []
        self.stride = stride  # stride of the starting frame

        # get image paths and load images
        img_nums = []
        vid_paths = []
        if isinstance(data_dir, str):
            # single dataset
            vid_names = sorted(os.listdir(data_dir))
            vid_paths = [opj(data_dir, vid_name) for vid_name in vid_names]
            if all(opif(vid_path) for vid_path in vid_paths):
                # data_dir is an image dir rather than a vid dir
                vid_paths = [data_dir]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                vid_names_n = sorted(os.listdir(data_dir_n))
                vid_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in vid_names_n]
                vid_paths.extend(vid_paths_n)

        for vid_path in vid_paths:
            img_names = sorted(os.listdir(vid_path))
            img_nums.append(len(img_names))
            self.img_paths.extend(
                [opj(vid_path, img_name) for img_name in img_names])

        # img_shape = None
        for img_path in tqdm(self.img_paths, desc='⏳ Loading dataset to Memory'):
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        counter = 0
        for img_num in img_nums:
            self.vid_idx.extend(
                list(range(counter, counter+img_num-self.vid_length+1, stride)))
            counter = counter+img_num

    def __getitem__(self, idx):
        # load video frames
        vid = self.imgs[self.vid_idx[idx]:self.vid_idx[idx]+self.vid_length]
        vid = np.array(vid)

        img_sz = vid[0].shape
        # crop to patch size
        if self.patch_sz:
            assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                        ), 'error PATCH_SZ larger than image size'
            xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])
            vid = vid[:, ymin:ymin+self.patch_sz[0],
                      xmin:xmin+self.patch_sz[1], :]
        # data augment
        if self.tform_op:
            vid = vid_transform(vid, tform_op=self.tform_op)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'

        if noise_level>0:
            image_dtype = vid.dtype
            image_maxv = np.iinfo(image_dtype).max  # 8/16 bit image -> 255/65535
            vid = vid + np.random.normal(0, image_maxv*noise_level, vid.shape)
            vid = vid.clip(0, image_maxv).astype(image_dtype)

        return vid.transpose(0, 3, 1, 2)

    def __len__(self):
        return len(self.vid_idx)


class Blurimg_RealExp_Dataset_all2CPU:
    """
    Dataset for real test: load real blurry image with no gt
    """

    def __init__(self, data_dir):
        super(Blurimg_RealExp_Dataset_all2CPU, self).__init__()
        self.data_dir = data_dir
        self.imgs = []

        # get blurry imag path
        if isinstance(data_dir, str):
            blur_names = sorted(os.listdir(data_dir))
            self.blur_paths = [opj(data_dir, blur_name)
                               for blur_name in blur_names]
        else:
            raise ValueError('data_dir should be a str')

        # load blurry image
        for img_path in tqdm(self.blur_paths, desc='⏳ Loading dataset to Memory'):
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx])
        return img.transpose(2, 0, 1)

    def __len__(self):
        return len(self.imgs)



# =================
# get dataloader
# =================

def get_data_loaders(data_dir, frame_num, batch_size, patch_size=None, tform_op=None, sigma_range=0, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    if status == 'train':
        if all2CPU:
            dataset = VideoFrame_Dataset_all2CPU(
                data_dir, frame_num, patch_size, tform_op, sigma_range)
        else:
            dataset = VideoFrame_Dataset(
                data_dir, frame_num, patch_size, tform_op, sigma_range)
    elif status == 'test':
        if all2CPU:
            dataset = VideoFrame_Dataset_all2CPU(
                data_dir, frame_num, patch_size, tform_op, sigma_range, frame_num)
        else:
            dataset = VideoFrame_Dataset(
                data_dir, frame_num, patch_size, tform_op, sigma_range, frame_num)
    elif status == 'real_test':
        dataset = Blurimg_RealExp_Dataset_all2CPU(
            data_dir)  # direct loading blurry image
    else:
        raise NotImplementedError(
            f"status ({status}) should be 'train' | 'test' ")

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    if status == 'train' or status == 'debug':
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        else:
            num_valid = int(num_total * validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
            DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)
