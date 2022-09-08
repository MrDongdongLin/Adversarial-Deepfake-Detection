import os
import argparse
import random


def parse_args():
    args_parser = argparse.ArgumentParser(description='Prepare train val test data of CelebDFv2 dataset')

    args_parser.add_argument('--list_test_txt', type=str, default='D:\\Deepfake\\Datasets\\Celeb-DF-v2\\List_of_testing_videos.txt')
    args_parser.add_argument('--images_path', type=str,
                             default='D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images')
    args_parser.add_argument('--output_txt', type=str, default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\CelebDFv2')
    args_parser.add_argument('--interval_frames', type=int, default=1)
    args_parser.add_argument('--num_frames', type=int, default=30)
    args_parser.add_argument('--random_seed', type=int, default=10)
    args = args_parser.parse_args()

    return args


def main():
    args = parse_args()
    list_test_txt = args.list_test_txt
    output_txt = args.output_txt
    images_path = args.images_path
    interval_frames = args.interval_frames
    num_frames = args.num_frames
    random_seed = args.random_seed

    # train+val+test = 400+82+108 = 590
    train_video_num = 400
    val_video_num = 82

    if not os.path.isdir(output_txt):
        os.mkdir(output_txt)

    celeb_all = []
    Celeb_real = os.path.join(images_path, 'Celeb-real')
    Celeb_synthesis = os.path.join(images_path, 'Celeb-synthesis')
    real_folder_list = os.listdir(Celeb_real)  # id9_0000, id9_0001
    fake_folder_list = os.listdir(Celeb_synthesis)  # id9_id0_0000

    test_list = []
    fin_test_videos = open(list_test_txt, 'r')
    for line in fin_test_videos:
        line = line.rstrip()
        words = line.split()  # ['1' 'Celeb-real/id9_0000.mp4']
        test_list.append((words[1].split('.')[0].replace('/', os.sep), int(words[0])))  # ['Celeb-real\\id9_0000' '1' ]
    fin_test_videos.close()
    print('Total videos include TEST set: REAL-{} FAKE-{}'.format(len(real_folder_list), len(fake_folder_list)))

    # ----------------------TEST---------------------
    test_total_real = 0
    test_total_fake = 0
    fout_test_list = open(os.path.join(output_txt, 'test.txt'), 'w')
    for video_path, label in test_list:
        f = 0
        video_dir = os.path.join(images_path, video_path)  # 'D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\Celeb-real\\id9_0000'

        video_name = video_dir.split(os.sep)[-1]  # 'id_0000'
        if video_name in real_folder_list:
            real_folder_list.remove(video_name)  # remove the test set
        if video_name in fake_folder_list:
            fake_folder_list.remove(video_name)

        img_list = os.listdir(video_dir)
        for i in range(0, len(img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(video_dir, img_list[i])
            line = img_path + ' ' + str(1-label) + '\n'  # real:0 fake:1
            fout_test_list.write(line)
            f += 1
            if 1-label:
                test_total_fake += 1
            else:
                test_total_real += 1
    fout_test_list.close()
    print('Total videos for TRAIN and VAL: REAL-{} FAKE-{}'.format(len(real_folder_list), len(fake_folder_list)))
    print()
    print('Test real-{}, fake-{}'.format(test_total_real, test_total_fake))

    random.seed(random_seed)
    random.shuffle(real_folder_list)
    random.seed(random_seed)
    random.shuffle(fake_folder_list)

    train_real = real_folder_list[:400]
    train_fake = fake_folder_list[:4300]
    val_real = real_folder_list[400-1:-1]
    val_fake = fake_folder_list[4300-1:-1]

    # ----------------------TRAIN---------------------
    fout_train_list = open(os.path.join(output_txt, 'train.txt'), 'w')
    t_id_res =[]
    train_total_fake = 0
    train_total_real = 0
    ct = 0
    for video_name in train_real:  # id9_0000
        # for real
        ct+=1
        real_video_dir = os.path.join(Celeb_real, video_name)  # D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\Celeb-real\\id9_0000
        real_img_list = os.listdir(real_video_dir)
        if len(real_img_list) < num_frames:
            # print(real_img_list, ct)
            continue
        f = 0
        for i in range(0, len(real_img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(real_video_dir, real_img_list[i])
            line = img_path + ' ' + str(0) + '\n'  # real:0
            fout_train_list.write(line)
            f += 1
            train_total_real += 1

        # corresponding fake TODO: since test set unpaired, the train set should be unpaired
        # id_num = video_name.split('_')[0]  # id9
        # id_res = [idx for idx in fake_folder_list if idx.startswith(id_num)]  # id9_id0_0000, id9_id0_0001, ...
        # if len(id_res) != 0:
        #     t_id_res = id_res
        # # print(len(id_res), video_name)
        # if len(id_res) == 0 and len(t_id_res) != 0:
        #     id_res = t_id_res  # let this time equals last time
        # random.seed(random_seed)
        # random.shuffle(id_res)
        # fake_video_dir = os.path.join(Celeb_synthesis, id_res[0])  # 'D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\Celeb-synthesis\\id12_id10_0006'

        fake_video_dir = os.path.join(Celeb_synthesis, train_fake[0])
        train_fake.remove(train_fake[0])  # when this one is chose, remove it
        fake_img_list = os.listdir(fake_video_dir)
        f = 0
        for i in range(0, len(fake_img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(fake_video_dir, fake_img_list[i])
            line = img_path + ' ' + str(1) + '\n'  # fake:1
            fout_train_list.write(line)
            f += 1
            train_total_fake += 1
    fout_train_list.close()
    print('Train real-{}, fake-{}'.format(train_total_real, train_total_fake))

    # ----------------------VAL---------------------
    t_id_res = []
    val_train_total_real = 0
    val_train_total_fake = 0
    fout_val_list = open(os.path.join(output_txt, 'val.txt'), 'w')
    for video_name in val_real:  # id9_0000
        # for real
        real_video_dir = os.path.join(Celeb_real, video_name)
        real_img_list = os.listdir(real_video_dir)
        if len(real_img_list) < num_frames: continue
        f = 0
        for i in range(0, len(real_img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(real_video_dir, real_img_list[i])
            line = img_path + ' ' + str(0) + '\n'  # real:0
            fout_val_list.write(line)
            f += 1
            val_train_total_real += 1

        # Also In Validation, we randomly choose the videos from fake to avoid overfiting
        # id_num = video_name.split('_')[0]
        # id_res = [idx for idx in fake_folder_list if idx.startswith(id_num)]
        # if len(id_res) != 0:
        #     t_id_res = id_res
        # # print(len(id_res), video_name)
        # if len(id_res) == 0 and len(t_id_res) != 0:
        #     id_res = t_id_res  # let this time equals last time
        # random.seed(random_seed)
        # random.shuffle(id_res)
        # fake_video_dir = os.path.join(Celeb_synthesis, id_res[0])

        fake_video_dir = os.path.join(Celeb_synthesis, val_fake[0])
        val_fake.remove(val_fake[0])
        fake_img_list = os.listdir(fake_video_dir)
        f = 0
        for i in range(0, len(fake_img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(fake_video_dir, fake_img_list[i])
            line = img_path + ' ' + str(1) + '\n'  # fake:1
            fout_val_list.write(line)
            f += 1
            val_train_total_fake += 1
    fout_val_list.close()
    print('Val real-{}, fake-{}'.format(val_train_total_real, val_train_total_fake))


if __name__ == "__main__":
    main()
