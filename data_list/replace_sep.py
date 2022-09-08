import os
import argparse


def replace_sep(args):
    infile_path = args.infile_path
    instr = args.instr
    replacestr = args.replacestr

    outfile_path = infile_path.replace("c40_val.txt", "linux_c40_val.txt")
    fin = open(infile_path, "r")
    fout = open(outfile_path, "w")
    for line in fin:
        new_line = line.replace(instr, replacestr)
        new_line = new_line.replace('\\', '/')
        fout.write(new_line)
    fin.close()
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_path', type=str, default='D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df\\c40_val.txt')
    parser.add_argument('--instr', type=str, default='D:\Deepfake\Datasets\FaceForensics++\mtcnn_with_bg')
    parser.add_argument('--replacestr', type=str, default='/media/hdddati1/donny/Datasets/FFpp/mtcnn_with_bg')
    args = parser.parse_args()
    replace_sep(args)

#
#
#
# root_path = "D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\four_df"
#
# data_cls = ["df", "f2f", "fs", "fsh", "nt", "c40_train", "c40_val", "c40_test"]
# data_class_dirs = os.listdir(root_path)
# for dc in data_cls:  # df, f2f, ...
#     for dcd in data_class_dirs:  # c40_df_train.txt, c40_train.txt, train.txt...
#         if dc in dcd:
#             data_path = os.path.join(root_path, dcd)  # ..random\\c40_df_train.txt
#             out_path = data_path.replace("c40_", "")
#             replace_sep(data_path)
#         if "c40_train" in dcd:
#             data_path = os.path.join(root_path, dcd)  # ..random\\c40_train.txt
#             out_path = data_path.replace("c40_", "")
#             replace_sep(data_path)
#         if "c40_val" in dcd:
#             data_path = os.path.join(root_path, dcd)
#             out_path = data_path.replace("c40_", "")
#             replace_sep(data_path)
#         if "c40_test" in dcd:
#             data_path = os.path.join(root_path, dcd)
#             out_path = data_path.replace("c40_", "")
#             replace_sep(data_path)
"""
infile_train = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\c40_train.txt", "r")
outfile_train = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\train.txt", "w")

for line in infile_train:
    new_line = line.replace("D:\\Deepfake\\Datasets\\FaceForensics++\\", "/media/hdddati1/donny/Faceforensics/")
    new_line = new_line.replace("\\", "/")
    # print(new_line)
    outfile_train.write(new_line)
infile_train.close()
outfile_train.close()

infile_val = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\c40_val.txt", "r")
outfile_val = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\val.txt", "w")

for line in infile_val:
    new_line = line.replace("D:\\Deepfake\\Datasets\\FaceForensics++\\", "/media/hdddati1/donny/Faceforensics/")
    new_line = new_line.replace("\\", "/")
    # print(new_line)
    outfile_val.write(new_line)
infile_val.close()
outfile_val.close()

infile_test = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\c40_test.txt", "r")
outfile_test = open("D:\\Deepfake\\Methods\\Deepfake-Detection-master\\data_list\\random\\test.txt", "w")

for line in infile_test:
    new_line = line.replace("D:\\Deepfake\\Datasets\\FaceForensics++\\", "/media/hdddati1/donny/Faceforensics/")
    new_line = new_line.replace("\\", "/")
    # print(new_line)
    outfile_test.write(new_line)
infile_test.close()
outfile_test.close()
"""
