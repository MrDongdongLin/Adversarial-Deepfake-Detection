"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import torch
import os


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        img_name = fn.split(os.sep)[-1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name

    def __len__(self):
        return len(self.imgs)


class MultiDataset(Dataset):
    def __init__(self, txt_path, frame_num, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        bs_imgs = list(self._split(imgs, frame_num))
        # delete overlayed frames like [0000011111]
        for i, bi in enumerate(bs_imgs):
            sum_labels = 0
            for t in bi:
                sum_labels += t[1]
            if sum_labels != 0 and sum_labels != frame_num:
                del bs_imgs[i]

        self.bs_imgs = bs_imgs
        self.transform = transform
        self.target_transform = target_transform

    def _split(self, list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    def __getitem__(self, index):
        bs_img = self.bs_imgs[index]
        inputs = []
        labels = []
        img_names = []
        for i in range(len(bs_img)):
            fn, label = bs_img[i]
            img = Image.open(fn).convert('RGB')
            img_name = fn.split(os.sep)[-1]

            if self.transform is not None:
                img = self.transform(img)

            inputs.append(img)
            labels = label
            img_names.append(img_name)

        return torch.stack(inputs), labels, img_names

    def __len__(self):
        return len(self.bs_imgs)
