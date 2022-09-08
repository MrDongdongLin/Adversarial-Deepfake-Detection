import torch
from torchvision import transforms
import cv2
import os
import numpy as np

import attacks.myfoolbox.foolbox as fb


def ifgsm_with_confidence(model, images, targets, confidence, epsilon, steps):
    images.requires_grad = True
    targets.requires_grad = False
    src_images = images.cpu().detach().numpy()
    src_targets = targets.cpu().detach().numpy()

    fmodel = fb.models.PyTorchModel(model, bounds=(-1, 1), num_classes=2)
    attack = fb.attacks.IterativeGradientSignAttack(fmodel)
    adv_images = attack(src_images, src_targets, confidence=confidence, epsilons=epsilon, steps=steps)

    return src_images, adv_images


def cw_with_confidence(model, images, targets, confidence):
    images.requires_grad = True
    targets.requires_grad = False
    src_images = images.cpu().detach().numpy()
    src_targets = targets.cpu().detach().numpy()

    fmodel = fb.models.PyTorchModel(model, bounds=(-1, 1), num_classes=2)
    attack = fb.attacks.CarliniWagnerL2Attack(fmodel)
    adv_images = attack(src_images, src_targets, confidence=confidence)

    return src_images, adv_images


def save_adv_images(adv_images, frame_ids, video_ids, data_dir, confidence, test_dataset_name, target_path):
    adv_image = torch.from_numpy(adv_images)
    mean = ([0.5] * 3)
    std = ([0.5] * 3)
    unorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    adv_image = unorm(adv_image).permute(1, 2, 0)
    # for adv, id in zip(adv_image, frame_ids):
    adv_img = adv_image.cpu().detach().numpy() * 255.0
    _adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    path_to_save = f'{data_dir}' + os.sep + f'{target_path}' + os.sep + f'{confidence}' + os.sep + f'{test_dataset_name}' + os.sep + f'{video_ids}' + os.sep
    image_name = f'{video_ids}_{frame_ids}.bmp'
    print(image_name)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        cv2.imwrite(f'{path_to_save}' + f'{image_name}', _adv_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(f'{path_to_save}' + f'{image_name}', _adv_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))