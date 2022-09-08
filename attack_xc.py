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
from attacks.myattacks import *


def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    is_attack = args.attack
    attack_method = args.attack_method
    save_images = args.save_images
    confidence = args.confidence
    epsilons = args.epsilons
    steps = args.steps

    torch.backends.cudnn.benchmark = True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                              num_workers=8)
    test_dataset_size = len(test_dataset)
    # model = torchvision.models.densenet121(num_classes=2)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()

    corrects = 0
    adv_corrects = 0
    val_real = 0.0
    adv_val_real = 0.0
    val_fake = 0.0
    adv_val_fake = 0.0
    val_real_size = 0
    val_fake_size = 0
    img_psnr = 0.0
    bs_num = 0
    valid_img_num = 0

    if attack_method == 'FGSM':
        path_to_save = os.path.join(args.path_to_save,
                                    'ifgsm_e' + str(epsilons) + 's' + str(steps) + 'c' + str(int(confidence)))
    elif attack_method == 'CW':
        path_to_save = os.path.join(args.path_to_save, 'cw_' + 'c' + str(int(confidence)))
    else:
        raise NotImplementedError
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    fout = open(os.path.join(path_to_save, 'path_attack_imgs.txt'), 'w')

    print("--------- ATTACK BEGINS ----------")
    for i, (image, labels, img_name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = image.cuda()
        labels = labels.cuda()
        if is_attack:
            if attack_method == 'FGSM':
                src_images, adv_images = ifgsm_with_confidence(model,
                                                               image,
                                                               labels,
                                                               confidence,
                                                               epsilon=epsilons,
                                                               steps=steps)
            elif attack_method == 'CW':
                src_images, adv_images = cw_with_confidence(model,
                                                            image,
                                                            labels,
                                                            confidence)
            else:
                raise NotImplementedError

            if save_images:
                # for i in range(batch_size):
                # TODO batch_size needs to be 1
                if np.isnan(adv_images).any() or (adv_images == src_images).all():
                    print("No Adv Found!!!")
                    continue
                else:
                    save_att_images(adv_images, path_to_save, img_name, labels, fout)

            adv_imgs = torch.from_numpy(adv_images).cuda()
            src_imgs = torch.from_numpy(src_images).cuda()
            psnr_bs = PSNR()(src_imgs, adv_imgs)
            if not torch.isinf(psnr_bs).any() and not torch.isnan(psnr_bs).any():
                img_psnr += psnr_bs
                valid_img_num += 1

            adv_outputs = model(adv_imgs)
            _, adv_preds = torch.max(adv_outputs.data, 1)
            adv_corrects += torch.sum(adv_preds == labels.data).to(torch.float32)
            adv_val_fake += torch.sum(adv_preds * labels.data).to(torch.float32)
            adv_val_real += torch.sum((1 - adv_preds) * (1 - labels.data)).to(torch.float32)

            outputs = model(src_imgs)
            _, preds = torch.max(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            val_fake += torch.sum(preds * labels.data).to(torch.float32)
            val_real += torch.sum((1 - preds) * (1 - labels.data)).to(torch.float32)

            val_fake_size += torch.sum(labels.data).to(torch.float32)
            val_real_size += torch.sum(1 - labels.data).to(torch.float32)

        bs_num += 1
    fout.close()

    acc = corrects / test_dataset_size
    adv_acc = adv_corrects / test_dataset_size
    avg_psnr = img_psnr / valid_img_num
    fake_acc = val_fake / val_fake_size
    real_acc = val_real / val_real_size
    adv_fake_acc = adv_val_fake / val_fake_size
    adv_real_acc = adv_val_real / val_real_size

    test_info = 'Test dataset size: ALL-{} REAL-{} FAKE-{}'.format(test_dataset_size, val_real_size, val_fake_size)
    test_acc = 'Test Avg Acc: {:.4f} - Real: {:.4f} - Fake: {:.4f}'.format(acc, real_acc, fake_acc)
    test_adv_acc = 'Test Avg Acc: {:.4f} - Real: {:.4f} - Fake: {:.4f}'.format(adv_acc, adv_real_acc, adv_fake_acc)
    test_psnr = 'Avg PSNR: {:.4f}'.format(avg_psnr)
    test_log = open(os.path.join(path_to_save, 'attack_info.txt'), 'w')
    test_log.write(test_info + '\n')
    test_log.write(test_acc + '\n')
    test_log.write(test_adv_acc + '\n')
    test_log.write(test_psnr + '\n')
    test_log.close()
    tqdm.write(test_info)
    tqdm.write(test_acc)
    tqdm.write(test_adv_acc)
    tqdm.write(test_psnr)



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=1)
    parse.add_argument('--test_list', '-tl', type=str,
                       default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\CelebDFv2\\test_attack.txt')
    parse.add_argument('--model_path', '-mp', type=str,
                       default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs\\xc_celebdfv2\\best.pt')
    parse.add_argument('--path_to_save', type=str, default='D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\attacks\\xc')
    parse.add_argument('--attack', action="store_true", help='Using adversarial attakcs')
    parse.add_argument('--attack_method', type=str, default='FGSM')
    parse.add_argument('--save_images', action='store_true')
    parse.add_argument('--confidence', type=float, default=10.0)
    parse.add_argument('--epsilons', type=int, default=15, help='epsilons in FGSM')
    parse.add_argument('--steps', type=int, default=40, help='steps in FGSM')
    main()
    # print('Hello world!!!')
