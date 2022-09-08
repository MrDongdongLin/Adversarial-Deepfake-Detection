python attack_xl.py \
--test_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/linux_test_attack.txt \
--model_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs/xl_celebdfv2/best.pt \
--path_to_save /media/ssddati1/donny/Datasets/images/attack/xl \
--attack \
--attack_method FGSM \
--save_images \
--confidence 10.0 \
--epsilons 15 \
--steps 40 \
--batch_size 2

python attack_xl.py \
--test_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/linux_test_attack.txt \
--model_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs/xl_celebdfv2/best.pt \
--path_to_save /media/ssddati1/donny/Datasets/images/attack/xl \
--attack \
--attack_method FGSM \
--save_images \
--confidence 1.0 \
--epsilons 15 \
--steps 40 \
--batch_size 2

python attack_xl.py \
--test_list /media/ssddati1/donny/Methods/Deepfake-Detection-master/data_list/CelebDFv2/linux_test_attack.txt \
--model_path /media/ssddati1/donny/Methods/Deepfake-Detection-master/outputs/xl_celebdfv2/best.pt \
--path_to_save /media/ssddati1/donny/Datasets/images/attack/xl \
--attack \
--attack_method FGSM \
--save_images \
--confidence 3.0 \
--epsilons 15 \
--steps 40 \
--batch_size 2
