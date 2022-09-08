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

    if not os.path.isdir(output_txt):
        os.mkdir(output_txt)

    test_list = []
    fin_test_videos = open(list_test_txt, 'r')
    for line in fin_test_videos:
        line = line.rstrip()
        words = line.split()  # ['1' 'Celeb-real/id9_0000.mp4']
        test_list.append((words[1].split('.')[0].replace('/', os.sep), int(words[0])))  # ['Celeb-real\\id9_0000' '1' ]
    fin_test_videos.close()
    # print('Total videos include TEST set: REAL-{} FAKE-{}'.format(len(real_folder_list), len(fake_folder_list)))

    # ----------------------TEST---------------------
    test_total_real = 0
    test_total_fake = 0
    real_num = 0  # 10 * num_frames
    fake_num = 0  # 10 * num_frames
    fout_test_list = open(os.path.join(output_txt, 'test_attack.txt'), 'w')
    for video_path, label in test_list:
        f = 0
        video_dir = os.path.join(images_path, video_path)  # 'D:\\Deepfake\\Datasets\\Celeb-DF-v2\\images\\Celeb-real\\id9_0000'

        img_list = os.listdir(video_dir)
        for i in range(0, len(img_list), interval_frames):
            if f >= num_frames:
                break
            img_path = os.path.join(video_dir, img_list[i])
            line = img_path + ' ' + str(1-label) + '\n'  # real:0 fake:1
            if real_num < 10*num_frames and 1-label==0:  # real < 600
                fout_test_list.write(line)
                real_num += 1
            elif fake_num < 10*num_frames and 1-label==1:
                fout_test_list.write(line)
                fake_num += 1
            else:
                continue
            f += 1
            if 1-label:
                test_total_fake += 1
            else:
                test_total_real += 1
    fout_test_list.close()
    # print('Total videos for TRAIN and VAL: REAL-{} FAKE-{}'.format(len(real_folder_list), len(fake_folder_list)))
    # print()
    print('Test real-{}, fake-{}'.format(test_total_real, test_total_fake))


if __name__ == "__main__":
    main()
