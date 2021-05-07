import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from ntu_read_skeleton_fortest import read_xyz
from ntu_read_skeleton_fortest import read_seqlen

#代表这些人的动作用于训练，其他人的动作用于验证，即Cross-Subject划分准则
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
#代表只用2号、3号相机产生的数据进行训练，1号相机采集的数据用于验证，即Cross-View划分准则
training_cameras = [2, 3]
#最多两个人
max_body = 2
#总共关节数为25
num_joint = 25
#最多300帧
max_frame = 300

'''
输出过程信息
'''
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


    
'''
data_path:数据存放路径

out_path:输出路径

ignored_sample_path:用于存放忽略文件名的地方，如果为空则代表不忽略文件

benchmark:('xview','xsub')
    若为xview则代表按照不同相机来划分训练集和验证集（即相机1采集的数据作为验证集，相机2、3采集的数据作为训练集）
    若为xsub则代表按照不同的人来划分训练集和验证集（即在training_subjects中的人对应的数据为训练集，其他的人对应的数据为验证集）
    
part:('eval','train')，为eval代表验证集，为train代表训练集
'''
def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    
    # step1.看是否有需要忽略的样本
    # ignored_sample_path中出现的文件名即代表忽略对应样本
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
        
    # step2.依次遍历数据集目录下的文件名，通过是否需要忽略样本以及样本是否符合当前状态（是在训练还是在验证）以将需要读取的样本以及标签加入sample_name和sample_label中    
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        # 如果该文件名字在忽略样本中，则忽略该文件
        if filename in ignored_samples:
            continue
        # 文件名A开始的部分代表动作的分类
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        # 文件名P开始的部分代表人物的id
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        # 文件名C开始的部分代表相机id
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        
        # 看基准是xview还是xsub，若为xview则代表按相机来划分训练集和验证集，则看当前文件的文件名中对应的相机是否为训练相机
        # 若为xsub则代表以人为基准，则看文件名中对应的人是否在训练集对应的人中
        # istraining用于代表是否为训练部分
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        # 看传入的模式是在训练还是在验证，若在训练则取istraining的值以看当前文件是否为训练样本
        # 若在验证则对istraining的值取反以看当前文件是否为验证样本
        # issample代表当前文件是否需要读取（采样）
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        # 如果当前文件需要读取，则加入到列表中
        # action_class为由文件名给出的ground_truth动作类别号
        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    # 将以上获得的需要读取的样本文件名以及ground_truth动作类别号存入out_path的train_label.pkl（训练集）或eval_label.pkl（验证集）中
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        # 将待取样的sample_name和sample_label以ascii形式存入f指向的文件中
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    # 再打开一个numpy类型的存数据文件流
    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    
    # 依次读取样本文件并写入
    # 数据的第0维是序号即代表是哪一个样本，第一维存的是关节的x,y,z坐标，第二维用于标识是哪一帧、第三维用于标识是哪个关节、第四维用于表示是哪个身体
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data,seq_len = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        # data:第一维存的是关节的x,y,z坐标，第二维用于标识是哪一帧、第三维用于标识是哪个关节、第四维用于表示是哪个身体
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()

#该函数为新增的，用于获取序列信息
def genseq_len(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    # step1.看是否有需要忽略的样本
    # ignored_sample_path中出现的文件名即代表忽略对应样本
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
        
    # step2.依次遍历数据集目录下的文件名，通过是否需要忽略样本以及样本是否符合当前状态（是在训练还是在验证）以将需要读取的样本以及标签加入sample_name和sample_label中    
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        # 如果该文件名字在忽略样本中，则忽略该文件
        if filename in ignored_samples:
            continue
        # 文件名A开始的部分代表动作的分类
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        # 文件名P开始的部分代表人物的id
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        # 文件名C开始的部分代表相机id
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        
        # 看基准是xview还是xsub，若为xview则代表按相机来划分训练集和验证集，则看当前文件的文件名中对应的相机是否为训练相机
        # 若为xsub则代表以人为基准，则看文件名中对应的人是否在训练集对应的人中
        # istraining用于代表是否为训练部分
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        # 看传入的模式是在训练还是在验证，若在训练则取istraining的值以看当前文件是否为训练样本
        # 若在验证则对istraining的值取反以看当前文件是否为验证样本
        # issample代表当前文件是否需要读取（采样）
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        # 如果当前文件需要读取，则加入到列表中
        # action_class为由文件名给出的ground_truth动作类别号
        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    #该文件流用于写骨架数据
    fp2 = open_memmap(
        '{}/{}_data_seqlen.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label),1))
    # 依次读取样本文件并写入
    # 数据的第0维是序号即代表是哪一个样本，第一维存的是关节的x,y,z坐标，第二维用于标识是哪一帧、第三维用于标识是哪个关节、第四维用于表示是哪个身体
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        seq_len = read_seqlen(os.path.join(data_path, s))
        # data:第一维存的是关节的x,y,z坐标，第二维用于标识是哪一帧、第三维用于标识是哪个关节、第四维用于表示是哪个身体
        #fp[i, :, 0:data.shape[1], :, :] = data
        print(seq_len)
        print('\n\n')
        fp2[i] = seq_len
    end_toolbar()
    
'''
为了只生成需要的序列信息数据，将其他数据删去！
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='D:/ntu-rgb+d/raw_txt')
    parser.add_argument(
        '--ignored_sample_path',
        default='D:/ntu-rgb+d/missing_list/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='D:/ntu-rgb+d/cooked_dat')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

'''    
# 只有当运行该文件时才执行，其中的参数需要改为自己数据集的位置
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='D:/ntu-rgb+d/raw_txt')
    parser.add_argument(
        '--ignored_sample_path',
        default='D:/ntu-rgb+d/missing_list/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='D:/ntu-rgb+d/cooked_dat')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            genseq_len(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
