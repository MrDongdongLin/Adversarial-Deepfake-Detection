## Train Xception

```shell
python3.9 train_xc.py \
--root_path D:\Deepfake\Methods\Deepfake-Detection-master\outputs \
--name xc_celebdfv2 \
--train_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\train.txt \
--val_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\val.txt \
--batch_size 16 \
--epoches 20 \
--model_name xc_celebdf.pkl \
--log_dir D:\Deepfake\Methods\Deepfake-Detection-master\log
```

## Train EfficientNet

```shell
python3.9 train_ef.py \
--root_path D:\Deepfake\Methods\Deepfake-Detection-master\outputs \
--name ef_celebdfv2 \
--train_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\train.txt \
--val_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\val.txt \
--batch_size 16 \
--epoches 20 \
--model_name ef_celebdf.pkl \
--log_dir D:\Deepfake\Methods\Deepfake-Detection-master\log
```

## Train XceptionLSTM

```shell
python3.9 train_xl.py \
--root_path D:\Deepfake\Methods\Deepfake-Detection-master\outputs \
--name xl_celebdfv2 \
--train_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\train.txt \
--val_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\val.txt \
--batch_size 2 \
--epoches 20 \
--model_name xl_celebdf.pkl \
--log_dir D:\Deepfake\Methods\Deepfake-Detection-master\log \
--frame_num 5

CUDA_VISIBLE_DEVICES=2 python train_xl.py \
--root_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs \
--name xl_celebdfv2 \
--train_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/unpiared/linux_train.txt \
--val_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/unpiared/linux_val.txt \
--batch_size 2 \
--epoches 20 \
--model_name xl_celebdf.pkl \
--log_dir /media/ssddati1/donny/Methods/Deepfake-Detection-master/log \
--frame_num 5
```

## Train EfficientLSTM

```shell
python3.9 train_el.py \
--root_path D:\Deepfake\Methods\Deepfake-Detection-master\outputs \
--name el_celebdfv2 \
--train_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\train.txt \
--val_list D:\Deepfake\Methods\Deepfake-Detection-master\data_list\CelebDFv2\val.txt \
--batch_size 2 \
--epoches 20 \
--model_name el_celebdf.pkl \
--log_dir D:\Deepfake\Methods\Deepfake-Detection-master\log
```

## Attack Xception

```shell
CUDA_VISIBLE_DEVICES=2 python attack_xc.py \
--test_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/linux_test_attack.txt \
--model_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs/xc_celebdfv2/best.pt \
--path_to_save /media/ssddati1/donny/Datasets/images/attack/xc \
--attack \
--attack_method FGSM \
--save_images \
--confidence 0.0 \
--epsilons 15 \
--steps 40 \
--batch_size 4
```

## Attack EfficientNet

```shell
CUDA_VISIBLE_DEVICES=2 python attack_ef.py \
--test_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/linux_test_attack.txt \
--model_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs/ef_celebdfv2/best.pt \
--path_to_save /media/ssddati1/donny/Datasets/images/attack/ef \
--attack \
--attack_method FGSM \
--save_images \
--confidence 0.0 \
--epsilons 15 \
--steps 40 \
--batch_size 4
```