import numpy as np
import os

'''
该函数用于读取骨架数据
'''
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        # 样本文件中第一行的值记录了总共有多少帧
        skeleton_sequence['numFrame'] = int(f.readline())
        
        # 以下为每一帧的信息
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            # 每一帧信息中的第一行为当前帧的身体数量信息
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            # 对当前帧每个身体依次获得当前的身体信息
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                # 每个当前帧信息的第三行 为关节点数目
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                # 依次获得每一个关节点的信息
                for v in range(body_info['numJoint']):
                    # joint_info的键
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                   
                    
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


'''
读取关节点数据
'''
# data:第一维存的是关节的x,y,z坐标，第二维用于标识是哪一帧、第三维用于标识是哪个关节、第四维用于表示是哪个身体
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    # 获取每个身体的每个关节点在每个帧上的xyz坐标
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    seq_len = np.zeros(1)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def read_seqlen(file):
    seq_info = read_skeleton(file)
    # 获取帧序列的长度
    seq_len = seq_info['numFrame']
    return seq_len