import time
import torch
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm

from dataset.transform import *


def print_training_info(batch_accuracy, loss, step, writer, epoch, train_log):
    log_info = '[Epoch {:03d} {}]: Iteration {} Training - Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, time.ctime(), step, loss.item(), batch_accuracy)
    train_log.write(log_info + "\n")
    tqdm.write(log_info)

    writer.add_scalar('training loss', loss.item(), step)
    writer.add_scalar('training acc', batch_accuracy, step)


def print_val_info(val_loss, val_accuracy, fake_acc, real_acc, epoch, writer, val_log):
    log_info = '[Epoch {:03d} {}]: Validation - Loss: {:.4f}, Accuracy: {:.4f}, Real: {:.4f}, Fake: {:.4f}'.format(epoch, time.ctime(), val_loss, val_accuracy, real_acc, fake_acc)
    val_log.write(log_info + "\n")
    tqdm.write(log_info)

    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation acc', val_accuracy, epoch)
    writer.add_scalar('real acc', fake_acc, epoch)
    writer.add_scalar('fake acc', real_acc, epoch)


def save_att_images(adv_images, path_to_save, image_name, labels, fout):
    for adv, name, lb in zip(adv_images, image_name, labels):

        _adv = torch.from_numpy(adv)
        lb = lb.cpu().numpy()
        # mean = ([0.5] * 3)
        # std = ([0.5] * 3)
        # unorm = transforms.Normalize(
        #     mean=[-m / s for m, s in zip(mean, std)],
        #     std=[1 / s for s in std]
        # )
        unorm = reverse_norm
        _adv = unorm(_adv).permute(1, 2, 0)
        # for adv, id in zip(adv_image, frame_ids):
        adv_img = _adv.cpu().detach().numpy() * 255.0
        _adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR).astype(np.uint8)

        # path_to_save = f'{data_dir}' + os.sep + f'{target_path}' + os.sep + f'{confidence}' + os.sep + f'{test_dataset_name}' + os.sep + f'{video_ids}' + os.sep
        # image_name = f'{video_ids}_{frame_ids}.bmp'
        img_path = os.sep.join([path_to_save, name])
        # print(name)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        cv2.imwrite(img_path, _adv_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        fout.write(img_path + ' ' + str(lb) + '\n')
