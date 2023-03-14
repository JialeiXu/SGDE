import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
cam_name_To_R = {'CAMERA_01':'CAMERA_06','CAMERA_06':'CAMERA_08',
            'CAMERA_08':'CAMERA_09','CAMERA_09':'CAMERA_07',
            'CAMERA_07':'CAMERA_05','CAMERA_05':'CAMERA_01'}
name_nuScenes_trans={'CAMERA_01':'F','CAMERA_06':'FR',
                     'CAMERA_08':'BR','CAMERA_09':'B',
                     'CAMERA_07':'BL','CAMERA_05':'FL'}
datasets = 'nuScenes' # or DDAD
for video_num in tqdm(range(5,30)):
    video_num = str('%06d'%video_num)
    save_path = '../../save/optimized_pose/nuScenes/6Camera_7Frame/LoFTR/USAC_ACCURATE/'+ video_num + '/'
    para_path = 'save_matching_points/nuScenes/6Camera_7Frame/USAC_ACCURATE/%s/parameters.txt'%video_num    # save_matching_points

    if not os.path.isfile(para_path):
        print('no parameters.txt', video_num)
        continue

    cam2num = {
        'CAMERA_01':[],
        'CAMERA_05':[],
        'CAMERA_06':[],
        'CAMERA_07':[],
        'CAMERA_08':[],
        'CAMERA_09':[]
    }
    cam2vehicle = {}
    para_file = open(para_path,'r')
    for i in range(36):
        line = para_file.readline()
        line = line.split(' ')[2][:-1]
        if i in range(0,6):
            cam2num['CAMERA_01'].append(float(line))
        if i in range(6,12):
            cam2num['CAMERA_05'].append(float(line))
        if i in range(12,18):
            cam2num['CAMERA_06'].append(float(line))
        if i in range(18,24):
            cam2num['CAMERA_07'].append(float(line))
        if i in range(24,30):
            cam2num['CAMERA_08'].append(float(line))
        if i in range(30,36):
            cam2num['CAMERA_09'].append(float(line))

    os.makedirs(save_path,exist_ok=True)

    for k,v in cam2num.items():
        t = []
        r = R.from_rotvec(v[:3])
        r = r.as_matrix()
        t.append([v[3]])
        t.append([v[4]])
        t.append([v[5]])

        file_name = save_path+'From_Cam_01To'+k+'.npz'

        #np.savez_compressed(file_name, R=r, T=t)
        ex = np.zeros((4,4))
        ex[:3,:3] = r
        ex[:3,3:] = t
        ex[3,3] = 1.0
        ex = np.linalg.pinv(ex) #vehicle2camera ->  camera2vehicle
        #print(k)
        file_name = save_path+k+'.npz'
        #print(file_name)
        #np.savez_compressed(file_name,BA_extrinsics=ex)
        cam2vehicle[k] = ex
        #print(ex)

    #here for stereo rectify, each camera compute ex to cam_r
    for k,v in cam_name_To_R.items():
        if datasets == 'nuScenes':
            file_name = save_path + name_nuScenes_trans[k]+'_ToCam_r.npz'
        elif datasets=='DDAD':
            file_name = save_path + k+'_ToCam_r.npz'
        else:
            raise ValueError('which datasets?')
        #print(file_name,cam_name_To_R[k])
        ex_lr = np.linalg.pinv(cam2vehicle[cam_name_To_R[k]]) @ cam2vehicle[k]
        #print(ex_lr)
        #print(vehicle2cam[k])
        np.savez_compressed(file_name, extrinsics=ex_lr)


print('done')
