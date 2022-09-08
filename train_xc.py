import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
from tqdm import tqdm

from utils import *
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset


def main():
    args = parse.parse_args()
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    root_path = args.root_path
    writer = SummaryWriter(comment='XC')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = os.path.join(root_path, name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark = True
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    # ])
    train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
    val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                             num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    if continue_train:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=1 / 4, patience=2, verbose=True
    # )
    # model = nn.DataParallel(model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0

    writer.add_hparams(args.__dict__, {})
    writer.add_text('model', str(model))

    train_log = open(os.path.join(args.log_dir, name + "_train_log.txt"), "w")
    val_log = open(os.path.join(args.log_dir, name + "_val_log.txt"), "w")
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        val_real = 0.0
        val_fake = 0.0
        val_real_size = 0
        val_fake_size = 0
        for i, (image, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 100):
                # print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
                print_training_info(iter_corrects / batch_size, loss, iteration, writer, epoch, train_log)
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            for i, (image, labels, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                image = image.cuda()
                labels = labels.cuda()
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
                val_fake += torch.sum(preds * labels.data).to(torch.float32)
                val_real += torch.sum((1 - preds) * (1 - labels.data)).to(torch.float32)
                val_fake_size += torch.sum(labels.data).to(torch.float32)
                val_real_size += torch.sum(1 - labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            epoch_fake_acc = val_fake / val_fake_size
            epoch_real_acc = val_real / val_real_size
            # print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            print_val_info(epoch_loss, epoch_acc, epoch_fake_acc, epoch_real_acc, epoch, writer, val_log)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        # if not (epoch % 40):
        torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_path, "best.pt"))
    train_log.close()
    val_log.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--root_path', '-r', type=str,
                       default="/media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs",
                       # default="D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs"
                       )
    parse.add_argument('--name', '-n', type=str, default='xc_c40_df_299')
    parse.add_argument('--train_list', '-tl', type=str,
                       default = '/media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/four_df/linux_c40_train.txt'
                       # default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df\\c40_train.txt'
                       )
    parse.add_argument('--val_list', '-vl', type=str,
                       default='/media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/four_df/linux_c40_val.txt',
                       # default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df\\c40_val.txt'
                       )
    parse.add_argument('--batch_size', '-bz', type=int, default=8)
    parse.add_argument('--epoches', '-e', type=int, default='20')
    parse.add_argument('--model_name', '-mn', type=str, default='xc_c40_f2f_299.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_c0_299.pkl')
    parse.add_argument('--log_dir', type=str,
                       # default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\log'
                       default='/media/ssddati1/donny/Methods/Deepfake-Detection-master/log'
                       )
    main()
