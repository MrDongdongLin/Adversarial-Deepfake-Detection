import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from tqdm import tqdm
import os
import cv2

from utils import *
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset


def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark = True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    # model = torchvision.models.densenet121(num_classes=2)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()
    val_real = 0.0
    val_fake = 0.0
    val_real_size = 0
    val_fake_size = 0
    with torch.no_grad():
        for i, (image, labels, img_name) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            val_fake += torch.sum(preds * labels.data).to(torch.float32)
            val_real += torch.sum((1 - preds) * (1 - labels.data)).to(torch.float32)
            val_fake_size += torch.sum(labels.data).to(torch.float32)
            val_real_size += torch.sum(1 - labels.data).to(torch.float32)
            # print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32) / batch_size))
            # print_val_info(epoch_loss, epoch_acc, epoch_fake_acc, epoch_real_acc, epoch, writer, val_log)
        acc = corrects / test_dataset_size
        fake_acc = val_fake / val_fake_size
        real_acc = val_real / val_real_size
        print('Test dataset size: ALL-{} REAL-{} FAKE-{}'.format(test_dataset_size, val_real_size, val_fake_size))
        print('Test Avg Acc: {:.4f} - Real: {:.4f} - Fake: {:.4f}'.format(acc, real_acc, fake_acc))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--test_list', '-tl', type=str,
                       default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df\\c40_test.txt')
    parse.add_argument('--model_path', '-mp', type=str,
                       default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs\\four_df\\best.pt')
    main()
    # print('Hello world!!!')
