python3.9 attack_el.py \
--test_list D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\CelebDFv2\\test_attack.txt \
--model_path D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs\\el_celebdfv2\\best.pt \
--path_to_save D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\attacks\\el \
--attack \
--attack_method FGSM \
--save_images \
--confidence 10.0 \
--epsilons 15 \
--steps 40 \
--batch_size 1

python3.9 attack_el.py \
--test_list D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\CelebDFv2\\test_attack.txt \
--model_path D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs\\el_celebdfv2\\best.pt \
--path_to_save D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\attacks\\el \
--attack \
--attack_method FGSM \
--save_images \
--confidence 1.0 \
--epsilons 15 \
--steps 40 \
--batch_size 1

python3.9 attack_el.py \
--test_list D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\CelebDFv2\\test_attack.txt \
--model_path D:\\Deepfake\\Methods\\Deepfake-Detection-master\\outputs\\el_celebdfv2\\best.pt \
--path_to_save D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\attacks\\el \
--attack \
--attack_method FGSM \
--save_images \
--confidence 3.0 \
--epsilons 15 \
--steps 40 \
--batch_size 1
