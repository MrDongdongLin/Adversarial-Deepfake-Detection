import os
import json


data_dir = "D:\\Deepfake\\Datasets\\FaceForensics++\\mtcnn_with_bg"
splits_path = "D:\\Deepfake\\Methods\\CNNRNN\\datasets\\Faceforensics++\\splits"
txt_outputs = "D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df"

data_classes = ["deepfakes", "face2face", "faceswap", "neural_textures", "original"]
compression = "c40"

if not os.path.exists(txt_outputs):
    os.makedirs(txt_outputs)

train_path = os.path.join(txt_outputs, compression + "_train.txt")
val_path = os.path.join(txt_outputs, compression + "_val.txt")
test_path = os.path.join(txt_outputs, compression + "_test.txt")

df_train = os.path.join(txt_outputs, compression + "_df_train.txt")
f2f_train = os.path.join(txt_outputs, compression + "_f2f_train.txt")
fs_train = os.path.join(txt_outputs, compression + "_fs_train.txt")
fsh_train = os.path.join(txt_outputs, compression + "_fsh_train.txt")
nt_train = os.path.join(txt_outputs, compression + "_nt_train.txt")

df_val = os.path.join(txt_outputs, compression + "_df_val.txt")
f2f_val = os.path.join(txt_outputs, compression + "_f2f_val.txt")
fs_val = os.path.join(txt_outputs, compression + "_fs_val.txt")
fsh_val = os.path.join(txt_outputs, compression + "_fsh_val.txt")
nt_val = os.path.join(txt_outputs, compression + "_nt_val.txt")

df_test = os.path.join(txt_outputs, compression + "_df_test.txt")
f2f_test = os.path.join(txt_outputs, compression + "_f2f_test.txt")
fs_test = os.path.join(txt_outputs, compression + "_fs_test.txt")
fsh_test = os.path.join(txt_outputs, compression + "_fsh_test.txt")
nt_test = os.path.join(txt_outputs, compression + "_nt_test.txt")

max_videos = 1000
train_size = 1000
num_frame_fake_train = 30
num_frame_real_train = 120  # if combine, real sampled N times for N fake
interval_frames = 1  # 隔5帧抽一帧
# num_frame_fake_val = 100
# num_frame_real_val = 50  # if combine, real sampled N times for N fake

fall = True  # if true, use all fakes, if not, use separately


def read_dataset_v2():
    train_dict = {}
    val_dict = {}
    test_dict = {}
    data_class_dirs = os.listdir(data_dir)
    for data_class_dir in data_class_dirs :
        if compression in data_class_dir:
            data_class_dir_path = os.path.join(data_dir, data_class_dir)
            # if original faces, target = 0, else target = 1
            target = 0 if 'original' in data_class_dir.lower() else 1

            for spl in ['train', 'val', 'test']:
                if spl == "train":
                    # train_path = os.path.join(txt_outputs, compression + "_train.txt")
                    train_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=False)
                    train_dict[data_class_dir] = train_files
                    if 'original' in data_class_dir:
                        train_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=True)
                        train_dict["combine_real"] = train_files
                elif spl == "val":
                    # val_path = os.path.join(txt_outputs, compression + "_val.txt")
                    val_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=False)
                    val_dict[data_class_dir] = val_files
                    if 'original' in data_class_dir:
                        val_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=True)
                        val_dict["combine_real"] = val_files
                else:
                    # test_path = os.path.join(txt_outputs, compression + "_test.txt")
                    test_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=False)
                    test_dict[data_class_dir] = test_files
                    if 'original' in data_class_dir:
                        test_files = output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real=True)
                        test_dict["combine_real"] = test_files

    if fall:
        # write txt files including all types of videos
        write_file_all(train_dict, train_path)
        write_file_all(val_dict, val_path)
        write_file_all(test_dict, test_path)
    else:
        # write txt files for each type of video
        write_file_each(train_dict, "deepfakes", df_train)
        write_file_each(train_dict, "face2face", f2f_train)
        write_file_each(train_dict, "faceswap", fs_train)
        write_file_each(train_dict, "faceshifter", fsh_train)
        write_file_each(train_dict, "neural_textures", nt_train)
    
        write_file_each(val_dict, "deepfakes", df_val)
        write_file_each(val_dict, "face2face", f2f_val)
        write_file_each(val_dict, "faceswap", fs_val)
        write_file_each(val_dict, "faceshifter", fsh_val)
        write_file_each(val_dict, "neural_textures", nt_val)
    
        write_file_each(test_dict, "deepfakes", df_test)
        write_file_each(test_dict, "face2face", f2f_test)
        write_file_each(test_dict, "faceswap", fs_test)
        write_file_each(test_dict, "faceshifter", fsh_test)
        write_file_each(test_dict, "neural_textures", nt_test)


def output_dict(spl, data_class_dir_path, data_class_dir, target, combine_real):
    video_ids = get_video_ids(spl, splits_path)
    # listdir_with_full_paths is used to get paths of each video (including faces)
    video_paths = listdir_with_full_paths(data_class_dir_path)
    # choose faces folders according to videos_path
    videos = [x for x in video_paths if get_file_name(x) in video_ids]

    # data_size = train_size if spl == "train" else max_videos
    out = output_lines(videos, target, data_class_dir, combine_real)
    return out


def output_lines(video_dirs, target, data_class_dir, combine_real):
    # f = open(txt_path, "w")
    outputs = []
    if combine_real:  # sample real and fake = 1:1
        num_frames = num_frame_real_train if data_class_dir == "original_faces_"+compression else num_frame_fake_train
    else:
        num_frames = num_frame_fake_train
    for video_dir in video_dirs[:max_videos]:
        f = 0
        img_list = os.listdir(video_dir)
        # if want a random behaviour, add it here
        for i in range(0, len(img_list), interval_frames):
            if not img_list[i].endswith('png'):
                continue
            if f >= num_frames:
                break
            label = target
            img_path = os.path.join(video_dir, img_list[i])
            line = img_path + ' ' + str(label) + '\n'
            # f.write(line)
            outputs.append(line)
            f += 1
    # f.close()
    return outputs


def write_file_all(dict, output_path):
    output_file = open(output_path, "w")
    for k, v in dict.items():
        for data_cls in data_classes:
            if data_cls in k:
                if "original" in k:
                    for line in dict["combine_real"]:
                        output_file.write(line)
                else:
                    for line in v:
                        output_file.write(line)
                    # for line in test_dict["original_faces_c40"]:
                    #     ftest.write(line)
    output_file.close()


def write_file_each(dict, video_type, output_path):
    output_file = open(output_path, "w")
    for k, v in dict.items():
        if video_type in k:  # only fake
            for line in v:
                output_file.write(line)
            for line in dict["original_faces_"+compression]:  # add paired real
                # if i >= num_frame_fake_train * video_num: break
                output_file.write(line)
    output_file.close()


def get_video_ids(spl, splits_path):
    json_name = read_json(os.path.join(splits_path, f'{spl}.json'))
    video_ids = get_sets(json_name)
    return video_ids


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_file_name(file_path):
    return file_path.split(os.sep)[-1]


if __name__ == '__main__':
    read_dataset_v2()