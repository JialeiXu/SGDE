import torch
import multiprocessing
from multiprocessing import Pool
import argparse
import time
from PIL import Image
import numpy as np
import cv2, os, json, glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from tools.RAFT_Stereo.demo import RAFT_STEREO
from utils import norm, gray_to_colormap, rotationMatrixToEulerAngles
import shutil


def rotate_depth(crop_shape,depth,Cam_origin,R,P):
    # rotate depth
    meshgrid = np.meshgrid(range(crop_shape[2], crop_shape[3] + 1), range(crop_shape[0], crop_shape[1] + 1),
                           indexing='xy')
    # [986, 1913, 6562, 6945]
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    ones = np.ones_like(id_coords[0])
    pix_coords = np.stack([id_coords[0].reshape(-1), id_coords[1].reshape(-1), ones.reshape(-1)])  # (3, 2354176)
    pix_coords *= depth.reshape(-1)
    pix_coords = Cam_origin['intrinsics'] @ np.linalg.pinv(R) @ np.linalg.pinv(P[:3, :3]) @ pix_coords
    depth_rotated = pix_coords[2].reshape(depth.shape[0], depth.shape[1])
    return depth_rotated


def project_CamR_depth(args,depth_l, K_l, K_r, R, T, mask_l):
    E = np.zeros((4, 4))
    E[3, 3] = 1.
    E[:3, :3] = R
    E[:3, 3] = T
    shape = depth_l.shape
    depth_l = depth_l.reshape(-1)
    meshgrid = np.meshgrid(range(shape[1]), range(shape[0]), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    ones = np.ones_like(id_coords[0]).reshape(-1)
    zeros = np.zeros_like(id_coords[0]).reshape(-1)
    #try
    if args.mask:
        try:
            ones = np.where(mask_l[:, :, 0].reshape(-1)==255, ones, zeros)  # 可能存在找不到overlap部分，由于之前代码逻辑，就不会加载mask
        except:
            pass

    pix_coords = np.stack([id_coords[0].reshape(-1), id_coords[1].reshape(-1), ones])
    points_3d = pix_coords * depth_l
    points_3d = np.linalg.pinv(K_l) @ points_3d
    points_4d = np.stack([points_3d[0], points_3d[1], points_3d[2], ones])
    points_4d = E @ points_4d
    cam_points = K_r @ points_4d[:3, :]
    cam_points = cam_points.swapaxes(1, 0)
    uv = cam_points[:, :2]
    z_c = cam_points[:, 2]
    uv = uv / np.clip(cam_points[:, 2:3], 1e-4, 200)
    in_view = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < shape[1]) & \
              (uv[:, 1] < shape[0]) & (z_c[:] > 1e-3)
    uv, z_c = np.around(uv[in_view]).astype(np.int16), z_c[in_view]
    uv[:,1] = np.clip(uv[:,1],0,shape[0]-1)
    uv[:,0] = np.clip(uv[:,0],0,shape[1]-1)
    depth = np.zeros((shape[0], shape[1]))
    depth[uv[:, 1], uv[:, 0]] = z_c
    depth = depth.reshape(shape[0], shape[1])
    return depth

def Stereo_rectify_shift_for_K(args,P1,P2,item,stage):
    if stage==1:
        if args.shift_json_path!='':
            file_shift = json.load(open(args.shift_json_path,'r'))
            add_x,add_y = file_shift[item['video_num']][item['Camera']]
        else:
            add_x,add_y = 0,0
        P1[0][2],P2[0][2] = P1[0][2]+args.rectify_shift_x + add_x, P2[0][2]+args.rectify_shift_x + add_x #+ args.rectify_shift_x_stage2
        P1[1][2],P2[1][2] = P1[1][2]+args.rectify_shift_y + add_y, P2[1][2]+args.rectify_shift_y + add_y #+ args.rectify_shift_y_stage2
    elif stage==2:
        P1[0][2], P2[0][2] = P1[0][2] + args.rectify_shift_x_stage2, P2[0][2] + args.rectify_shift_x_stage2
        P1[1][2], P2[1][2] = P1[1][2] + args.rectify_shift_y_stage2, P2[1][2] + args.rectify_shift_y_stage2
    elif stage==3:
        #并不存在第三阶段，而是在back_stereo_rectify中集成了两个shift
        if args.shift_json_path !='':
            file_shift = json.load(open(args.shift_json_path, 'r'))
            add_x, add_y = file_shift[item['video_num']][item['Camera']]
        else:
            add_x,add_y = 0,0
        P1[0][2], P2[0][2] = P1[0][2] + args.rectify_shift_x + add_x + args.rectify_shift_x_stage2, P2[0][
            2] + args.rectify_shift_x + add_x  + args.rectify_shift_x_stage2
        P1[1][2], P2[1][2] = P1[1][2] + args.rectify_shift_y + add_y + args.rectify_shift_y_stage2, P2[1][
            2] + args.rectify_shift_y + add_y  + args.rectify_shift_y_stage2
    else:
        raise ValueError('Stage error')
    return P1, P2
def stereo_rectify_crop(img,crop):
    if crop==None:
        return img
    img = img[crop[0]:crop[1]+1 ,crop[2]:crop[3]+1]
    return img
def decide_crop_shape(img_l,img_r, rectify_size):
    overlap_l = img_l[:, :, 2] > 0
    overlap_r = img_r[:, :, 2] > 0
    overlap = np.logical_and(overlap_l, overlap_r)
    if overlap.max()==False: #没有overlap
        return [2**15-1,-1,2**15-1,-1], None
    h_d_l_r = [2**15-1,-1,2**15-1,-1]
    #————————————————————————————————————————————————————————
    overlap_corr = np.where(overlap)
    h_d_l_r[0] = overlap_corr[0].min()
    h_d_l_r[1] = overlap_corr[0].max()
    h_d_l_r[2] = overlap_corr[1].min()
    h_d_l_r[3] = overlap_corr[1].max()

    #————————————————————————————————————————————————————————————————————————————

    x =  0 if (h_d_l_r[1]-h_d_l_r[0]+1) % 32==0 else 1
    x = (x + (h_d_l_r[1]-h_d_l_r[0]+1)//32) * 32
    y = 0 if (h_d_l_r[3] - h_d_l_r[2]+1) % 32==0 else 1
    y = (y + (h_d_l_r[3]-h_d_l_r[2]+1)//32) * 32
    assert h_d_l_r[1]-h_d_l_r[0]<=x and h_d_l_r[3]-h_d_l_r[2]<=y
    h_add = x - (h_d_l_r[1] - h_d_l_r[0]+1)
    w_add = y - (h_d_l_r[3] - h_d_l_r[2]+1)
    h_d_l_r[0] -= h_add//2 + h_add%2
    h_d_l_r[1] += h_add//2
    h_d_l_r[2] -= w_add//2 + w_add%2
    h_d_l_r[3] += w_add//2

    if h_d_l_r[0]<0:
        h_d_l_r[1]-=h_d_l_r[0]
        h_d_l_r[0]=0
    if h_d_l_r[1] >= rectify_size[1]:
        h_d_l_r[0] -= h_d_l_r[1] - rectify_size[1]+1
        h_d_l_r[1] = rectify_size[1]-1
    if h_d_l_r[2]<0:
        h_d_l_r[3]-=h_d_l_r[2]
        h_d_l_r[2]=0
    if h_d_l_r[3]>=rectify_size[0]:
        h_d_l_r[2] -= h_d_l_r[3]-rectify_size[0]+1
        h_d_l_r[3] = rectify_size[0]-1
    assert h_d_l_r[0]>=0 and h_d_l_r[1]<rectify_size[1] and h_d_l_r[2]>=0 and h_d_l_r[3]<rectify_size[0]

    overlap_area = overlap[h_d_l_r[0]:h_d_l_r[1] + 1, h_d_l_r[2]:h_d_l_r[3] + 1]

    return h_d_l_r, overlap_area


def Stereo_Rectify(args,Cam_L,Cam_R,path,item,rgb_shape=(1936,1216), s=4,crop_shape=None):
    timestamp = str(item['timestamp'])
    Cam = item['Camera']

    img = cv2.resize(Cam_L['rgb'],rgb_shape)
    img_L = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(Cam_R['rgb'],rgb_shape)
    img_R = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #ex_str = args.ex_str#ex_str = 'pose' if False else 'extrinsics'
    ex_L = Cam_L[args.ex_str]
    ex_R = Cam_R[args.ex_str]

    ex = np.linalg.pinv(ex_R) @ ex_L
    R,T = ex[:3,:3],ex[:3,3]

    if args.optimize_pose:
        if args.specific_pose!='':
            ex = np.load(args.specific_pose + Cam + '_ToCam_r.npz')
            R, T = ex['extrinsics'][:3, :3], ex['extrinsics'][:3, 3]
        elif args.BA_6cam_pose!='':
            if os.path.isfile(args.BA_6cam_pose+Cam+'_ToCam_r.npz'):
                ex = np.load(args.BA_6cam_pose+Cam+'_ToCam_r.npz')
                R, T = ex['extrinsics'][:3, :3], ex['extrinsics'][:3, 3]
            #else 就使用GT 针对与final的pose

    dist_coefs = np.array([0, 0, 0., 0., 0])
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(Cam_L['intrinsics'], dist_coefs, Cam_R['intrinsics'], dist_coefs, rgb_shape, R, T, alpha=-1)
    out_shape = (rgb_shape[0] * s, rgb_shape[1] * s)
    out_shape_stage2 = (rgb_shape[0]*args.s_stage2, rgb_shape[1]*args.s_stage2)
    #stereo rectify stage 1:
    if crop_shape==None:
        assert args.rectify_shift_x_stage2==0 and args.rectify_shift_y_stage2==0
        P1, P2 = Stereo_rectify_shift_for_K(args, P1, P2, item,stage=1)
        mapL1, mapL2 = cv2.initUndistortRectifyMap(Cam_L['intrinsics'], dist_coefs, R1, P1, out_shape, cv2.CV_32FC1)  # cv2.CV_16SC2)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(Cam_R['intrinsics'], dist_coefs, R2, P2, out_shape, cv2.CV_32FC1)  # cv2.CV_16SC2)
        rectL = cv2.remap(img_L, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(img_R, mapR1, mapR2, cv2.INTER_LINEAR)
        crop_shape, overlap = decide_crop_shape(rectL, rectR, out_shape)
        if crop_shape != [2 ** 15 - 1, -1, 2 ** 15 - 1, -1]:
            args.rectify_shift_x_stage2, args.rectify_shift_y_stage2 = -crop_shape[2], -crop_shape[0]
            #args.rectify_shift_x_stage2, args.rectify_shift_y_stage2 = crop_shape[2], crop_shape[0]
            crop_shape[1]-=crop_shape[0]
            crop_shape[0]-=crop_shape[0]
            crop_shape[3]-=crop_shape[2]
            crop_shape[2]-=crop_shape[2]
            stage=2
    else:
        stage=3

    #stereo rectify stage 2:
    if crop_shape!= [2**15-1,-1,2**15-1,-1]:
        assert args.rectify_shift_x_stage2!=0 and args.rectify_shift_y_stage2!=0,(args.rectify_shift_x_stage2, args.rectify_shift_y_stage2)
        P1, P2 = Stereo_rectify_shift_for_K(args,P1,P2,item,stage=stage)

        mapL1, mapL2 = cv2.initUndistortRectifyMap(Cam_L['intrinsics'], dist_coefs, R1, P1, out_shape_stage2, cv2.CV_32FC1)#cv2.CV_16SC2)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(Cam_R['intrinsics'], dist_coefs, R2, P2, out_shape_stage2, cv2.CV_32FC1)#cv2.CV_16SC2)

        rectL = cv2.remap(img_L, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(img_R, mapR1, mapR2, cv2.INTER_LINEAR)
    if not args.crop: #控制crop
        crop_shape=None

    os.makedirs(path+'l/',exist_ok=True)
    os.makedirs(path+'r/',exist_ok=True)
    if args.mask:
        if args.mask_l_Rect_crop is None:
            mask_l = cv2.imread(args.mask_path + item['Camera'] + '_' + item['scene'] + '.png')
            mask_r = cv2.imread(args.mask_path + item['Camera_r'] + '_' + item['scene'] + '.png')
            mask_l = cv2.resize(mask_l, (rgb_shape[0], rgb_shape[1]))#[:,:,0]
            mask_r = cv2.resize(mask_r, (rgb_shape[0], rgb_shape[1]))#[:,:,0]
            mask_l_Rect = cv2.remap(mask_l, mapL1, mapL2, cv2.INTER_LINEAR)
            mask_r_Rect = cv2.remap(mask_r, mapR1, mapR2, cv2.INTER_LINEAR)
            mask_l_Rect_crop = stereo_rectify_crop(mask_l_Rect, crop_shape)
            mask_r_Rect_crop = stereo_rectify_crop(mask_r_Rect, crop_shape)
            args.mask_l_Rect_crop = mask_l_Rect_crop
            args.mask_r_Rect_crop = mask_r_Rect_crop
    #判断是否有overlap
    if crop_shape == [2**15-1,-1,2**15-1,-1]: #no overlap
        #with open(args.log,'a') as f:
        #    f.write(timestamp+'_'+Cam+'.npy'+'\n')
        cv2.imwrite(path + 'l/' + timestamp + '_' + Cam + '.png', np.zeros((rgb_shape[1],rgb_shape[0],3))) # debug
        cv2.imwrite(path + 'r/' + timestamp + '_' + Cam + '.png', np.zeros((rgb_shape[1],rgb_shape[0],3)))  # debug
    else:
        rectL_crop = stereo_rectify_crop(rectL,crop_shape)
        rectR_crop = stereo_rectify_crop(rectR,crop_shape)
        if args.mask:
            rectL_crop = np.where(args.mask_l_Rect_crop == 255, rectL_crop, np.zeros_like(args.mask_l_Rect_crop))  # debug
            rectR_crop = np.where(args.mask_r_Rect_crop == 255, rectR_crop, np.zeros_like(args.mask_r_Rect_crop))
        cv2.imwrite(path+'l/'+timestamp+ '_' + Cam +'.png', rectL_crop)
        cv2.imwrite(path+'r/'+timestamp+ '_' + Cam +'.png', rectR_crop)

        if True: #l+r
            os.makedirs(path + 'l_r/', exist_ok=True)
            #rect_sum = rectL_crop + rectR_crop
            rect_sum = cv2.addWeighted(rectL_crop,0.8,rectR_crop,0.8,0)
            cv2.imwrite(path + 'l_r/'+timestamp+'_'+Cam+'.png', rect_sum)
    return crop_shape

def filling(img,shape,location):
    img_big = np.zeros(shape)
    img_big[location[0]:location[1]+1 ,location[2]:location[3]+1] = img
    return img_big

def Back_Stereo_Rectify(args,Cam_L,Cam_R,item,disp_path,depth_path,depth_visualization_path,rgb_shape=(1936,1216), s=4,crop_shape=None):
    os.makedirs(depth_path,exist_ok=True)
    timestamp = str(item['timestamp'])
    Cam = item['Camera']

    #ex_str = arga.ex_str #ex_str = 'pose' if False else 'extrinsics'
    ex_L = Cam_L[args.ex_str]
    ex_R = Cam_R[args.ex_str]

    ex = np.linalg.pinv(ex_R) @ ex_L
    R, T = ex[:3, :3], ex[:3, 3]
    if args.optimize_pose:
        if args.specific_pose!='':
            ex = np.load(args.specific_pose + Cam + '_ToCam_r.npz')
            R, T = ex['extrinsics'][:3, :3], ex['extrinsics'][:3, 3]
        elif args.BA_6cam_pose!='':
            if os.path.isfile(args.BA_6cam_pose+Cam+'_ToCam_r.npz'):
                ex = np.load(args.BA_6cam_pose + Cam + '_ToCam_r.npz')
                R, T = ex['extrinsics'][:3, :3], ex['extrinsics'][:3, 3]

    dist_coefs = np.array([0, 0, 0., 0., 0])
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(Cam_L['intrinsics'], dist_coefs, Cam_R['intrinsics'], dist_coefs, rgb_shape, R, T, alpha=-1)
    P1, P2 = Stereo_rectify_shift_for_K(args,P1,P2,item,stage=3)
    ### Disp2Depth ###
    f = P1[0][0]
    baseline = P2[0][3]/f
    #baseline = -np.sqrt(T[0]*T[0]+T[1]*T[1]+T[2]*T[2])
    disp = np.load(disp_path+timestamp+'_'+Cam+'.npy')
    depth = f * baseline / disp
    if args.flip:
        disp_r = np.load(disp_path+'flip/'+timestamp+'_'+Cam+'.npy')
        disp_r = disp_r[:,::-1].copy()
        depth_r = f * baseline / disp_r

    img = cv2.resize(Cam_L['rgb'], rgb_shape)
    img_L = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(Cam_R['rgb'], rgb_shape)
    img_R = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_shape = (rgb_shape[0], rgb_shape[1])
    mapL1, mapL2 = cv2.initUndistortRectifyMap(Cam_L['intrinsics'], dist_coefs, R1, P1, out_shape,
                                               cv2.CV_32FC1)  # cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(Cam_R['intrinsics'], dist_coefs, R2, P2, out_shape,
                                               cv2.CV_32FC1)  # cv2.CV_16SC2)
    if crop_shape==None:
        rectL = cv2.remap(img_L, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(img_R, mapR1, mapR2, cv2.INTER_LINEAR)
        crop_shape, overlap = decide_crop_shape(rectL, rectR, out_shape)
    ####### back remap
    #out_shape = (rgb_shape[0]*args.s_stage2, rgb_shape[1]*args.s_stage2)
    mapL1_back, mapL2_back = cv2.initUndistortRectifyMap(P1[:3,:3],dist_coefs,np.linalg.pinv(R1),Cam_L['intrinsics'],out_shape,cv2.CV_32FC1)
    mapR1_back, mapR2_back = cv2.initUndistortRectifyMap(P2[:3,:3],dist_coefs,np.linalg.pinv(R2),Cam_R['intrinsics'],out_shape,cv2.CV_32FC1)
    #img_r 直接从back_remap_depth_l 投影，不从这里走了
    if crop_shape != [2**15-1,-1,2**15-1,-1] and True: #存在overlap区域，旋转不同相机下的深度
        depth_l = rotate_depth(crop_shape,depth,Cam_L,R1,P1)
        if args.flip:
            depth_r = rotate_depth(crop_shape,depth_r,Cam_R,R2,P2)
    else: depth_l = depth
    mask_l = None #for project_camR_depth
    #判断是否有overlap
    if crop_shape == [2**15-1,-1,2**15-1,-1]: #or timestamp=='1569371029886782': 历史遗留bug
        back_remap_depth_l = np.zeros((rgb_shape[1],rgb_shape[0]))
        back_remap_depth_r = np.zeros((rgb_shape[1],rgb_shape[0]))
    else:
        if args.mask:
            if args.mask_l_Rect_crop is None:
                mask_l = cv2.imread(args.mask_path + item['Camera'] + '_' + item['scene'] + '.png')
                mask_r = cv2.imread(args.mask_path + item['Camera_r'] + '_' + item['scene'] + '.png')
                mask_l = cv2.resize(mask_l, (rgb_shape[0], rgb_shape[1]))
                mask_r = cv2.resize(mask_r, (rgb_shape[0], rgb_shape[1]))
                # mask_Rect
                mask_l_Rect = cv2.remap(mask_l, mapL1, mapL2, cv2.INTER_LINEAR)
                mask_r_Rect = cv2.remap(mask_r, mapR1, mapR2, cv2.INTER_LINEAR)

                mask_l_Rect_crop = stereo_rectify_crop(mask_l_Rect, crop_shape)
                mask_r_Rect_crop = stereo_rectify_crop(mask_r_Rect, crop_shape)
                args.mask_l_Rect_crop = mask_l_Rect_crop
                args.mask_r_Rect_crop = mask_r_Rect_crop
        depth_l = np.where(args.mask_l_Rect_crop[:,:,0] == 255, depth_l, np.zeros_like(args.mask_l_Rect_crop[:,:,0]))
        depth_l = np.where(args.mask_r_Rect_crop[:,:,0] == 255, depth_l, np.zeros_like(args.mask_r_Rect_crop[:,:,0]))
        depth_l = filling(depth_l,(rgb_shape[1]*args.s_stage2,rgb_shape[0]*args.s_stage2), crop_shape)
        if args.flip:
            depth_r = np.where(args.mask_l_Rect_crop[:,:,0] == 255, depth_r, np.zeros_like(args.mask_l_Rect_crop[:,:,0]))
            depth_r = np.where(args.mask_r_Rect_crop[:,:,0] == 255, depth_r, np.zeros_like(args.mask_r_Rect_crop[:,:,0]))
            depth_r = filling(depth_r,(rgb_shape[1]*args.s_stage2,rgb_shape[0]*args.s_stage2), crop_shape)

        back_remap_depth_l = cv2.remap(depth_l, mapL1_back, mapL2_back, cv2.INTER_LINEAR)
        if args.flip:
            back_remap_depth_r = cv2.remap(depth_r,mapR1_back,mapR2_back,cv2.INTER_LINEAR)
    #    if args.mask:  集成到上面mask_rect了
    #        back_remap_depth_l = np.where(mask_l[:, :, 0] > args.eps, back_remap_depth_l, mask_l[:, :, 0])
    #clip
    back_remap_depth_l = np.clip(back_remap_depth_l,args.min_depth,args.max_depth)
    if args.flip:
        back_remap_depth_r = np.clip(back_remap_depth_r,args.min_depth,args.max_depth)

    # compute cam_r depth
    if not args.flip:
        back_remap_depth_r = project_CamR_depth(args,back_remap_depth_l, Cam_L['intrinsics'], Cam_R['intrinsics'], R, T, mask_l)
        if args.mask:
            mask_r = cv2.imread(args.mask_path + item['Camera_r'] + '_' + item['scene'] + '.png')
            mask_r = cv2.resize(mask_r, (rgb_shape[0], rgb_shape[1]))
            back_remap_depth_r = np.where(mask_r[:, :, 0] == 255, back_remap_depth_r, np.zeros_like(back_remap_depth_r))  # 存在找不到overlap情况，就不会加载mask
    back_remap_depth_l_img = Image.fromarray((back_remap_depth_l*200).astype(np.uint16))
    back_remap_depth_r_img = Image.fromarray((back_remap_depth_r*200).astype(np.uint16))
    back_remap_depth_l_img.save(depth_path+timestamp+'_'+Cam+'_l'+'.png')
    back_remap_depth_r_img.save(depth_path+timestamp+'_'+Cam+'_r'+'.png')
    #np.save(depth_path+timestamp+'_'+Cam+'_l',back_remap_depth_l)
    #np.save(depth_path+timestamp+'_'+Cam+'_r',back_remap_depth_r)
    if True:     #visualization depth
        os.makedirs(depth_visualization_path,exist_ok=True)
        if True:
            back_remap_depth_l = np.clip(back_remap_depth_l,0,100)
            depth_color_l = gray_to_colormap(back_remap_depth_l, max=100)
            depth_color_l = cv2.cvtColor(depth_color_l, cv2.COLOR_BGR2RGB)
            cv2.imwrite(depth_visualization_path+timestamp+'_'+Cam+'_l.png',depth_color_l)

            back_remap_depth_r = np.clip(back_remap_depth_r,0,100)
            depth_color_r = gray_to_colormap(back_remap_depth_r,max=100)
            depth_color_r = cv2.cvtColor(depth_color_r, cv2.COLOR_BGR2RGB)
            cv2.imwrite(depth_visualization_path+timestamp+'_'+Cam+'_r.png',depth_color_r)

        else:
            back_remap_depth_l = np.clip(back_remap_depth_l,0,200)
            back_remap_depth_l = (back_remap_depth_l-back_remap_depth_l.min())/(back_remap_depth_l.max()-back_remap_depth.min())*255
            depth_color = cv2.applyColorMap(back_remap_depth_l.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite(depth_visualization_path+timestamp+'_'+Cam+'.png',depth_color)

    return crop_shape

def disp2depth_Stereo(disp_path,save_depth_path,baseline,f):
    os.makedirs(save_depth_path,exist_ok=True)
    disp_list = sorted(glob.glob(disp_path))
    disp_len = len(disp_list)
    for i in range(disp_len):
        disp = np.load(disp_list[i])
        depth = f * baseline / disp
        np.save(save_depth_path + disp_list[i].split('/')[-1],depth)

    print('done')

def main_work(rank,args):
    args.rank = rank
    args.datasets_path+=args.train_val+'/'
    args.save_path+=args.name+'/'
    args.tmp_save_path = args.save_path

    os.makedirs(args.save_path,exist_ok=True)

    print(str(rank % torch.cuda.device_count()))
    #shutil.rmtree(args.save_path)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % torch.cuda.device_count())
    print('gpu=',rank % torch.cuda.device_count())
    torch.cuda.set_device(rank % torch.cuda.device_count())

    if len(args.video_list)==2:
        args.video_list = ['%06d' % i for i in range(int(args.video_list[0]), int(args.video_list[1]))]

    if os.path.exists(args.log):
        os.remove(args.log)
    json_file = json.load(open(args.json_file_path))

    for video_i in range(args.rank, len(args.video_list), args.processes_num):
        print('video_i',video_i)
        for Cam_i in range(len(args.Cam_list)):
            if args.mask:
                args.mask_l_Rect_crop = None
                args.mask_r_Rect_crop = None
            sequence_len = -1
            video, Cam = args.video_list[video_i],args.Cam_list[Cam_i]
            args.save_path = args.tmp_save_path + str('%06d'%int(video)) + '/'
            args.BA_6cam_pose = args.BA_6cam_pose_s%str(video) if args.BA_6cam_pose_s!='' else ''
            if args.specific_video_cam_JsonPath!='':
                specific_video_cam_Josn = json.load(open(args.specific_video_cam_JsonPath,'r'))

            print(Cam, 'ing...')
            dict_i = []
            for i in range(len(json_file[args.train_val])):
                if json_file[args.train_val][i]['video_num'] in video and json_file[args.train_val][i]['Camera'] in Cam:
                    sequence_len+=1
                    if args.specific_scene!='':
                        if json_file[args.train_val][i]['scene']!=args.specific_scene: break
                    if args.specific_video_cam_JsonPath!='':
                        decide_run_this_videl_cam=False
                        for specific_video_cam_Josn_i in range(len(specific_video_cam_Josn)):
                            if specific_video_cam_Josn[specific_video_cam_Josn_i][0]==video and specific_video_cam_Josn[specific_video_cam_Josn_i][1]==Cam:
                                decide_run_this_videl_cam=True
                                break
                        if decide_run_this_videl_cam==False: break
                    if sequence_len % args.skip_step!=0:    continue
                    if sequence_len < args.sequence_len:    dict_i.append(json_file[args.train_val][i])
            if len(dict_i)==0: continue
            crop_shape = None
            args.rectify_shift_x_stage2, args.rectify_shift_y_stage2 = 0,0
            print(video,Cam,len(dict_i),'this')

            #step 1
            #print('step 1. Stereo rectify and crop the overlap.')
            time_step1 = time.time()

            for i in range(len(dict_i)):
                #break# debug
                item = dict_i[i]
                Cam_L = np.load(args.datasets_path+str(item['timestamp'])+'_'+item['Camera']+'.npz')
                Cam_R = np.load(args.datasets_path+str(item['timestamp'])+'_'+item['Camera_r']+'.npz')
                crop_shape = Stereo_Rectify(
                    args,Cam_L,Cam_R,path=args.save_path,item=item,crop_shape=crop_shape,s=args.s)
                #crop_shape = None #debug 每个record都会重新计算crop_shape
            #print('step 1 done')
            print('step1=', time.time() - time_step1)
            time_step2 = time.time()
            #
            #step 2
            #print('step 2. calculate the disparity by RAFT STEREO')
            l_img, r_img = [], []
            for i in range(len(dict_i)):
                item = dict_i[i]
                l_img.append(args.save_path + 'l/'+str(item['timestamp'])+'_'+item['Camera']+'.png')
                r_img.append(args.save_path + 'r/'+str(item['timestamp'])+'_'+item['Camera']+'.png')
            RAFT_STEREO(l = l_img, # save_path + 'l/*.png',
                        r = r_img, #save_path + 'r/*.png',
                        output = args.save_path + 'disp')
            if args.flip:
                RAFT_STEREO(l = l_img, r = r_img, output = args.save_path + 'disp/flip',flip=args.flip)
            print('step2=',time.time()-time_step2)
            time_step3 = time.time()
            #print('step 2 done')

            #step 3
            disp_path = args.save_path + 'disp/'
            depth_path = args.save_path + 'depth/'
            depth_visualization_path = args.save_path + 'depth_visualization/'

            #crop_shape = None
            #print('step 3. Calculate the depth by disparity, remap back to original image')
            for i in range(len(dict_i)):
                item = dict_i[i]
                Cam_L = np.load(args.datasets_path + str(item['timestamp']) + '_' + item['Camera'] + '.npz')
                Cam_R = np.load(args.datasets_path + str(item['timestamp']) + '_' + item['Camera_r'] + '.npz')
                crop_shape = Back_Stereo_Rectify(args,
                                                 Cam_L, Cam_R, item, disp_path, depth_path,
                                                 depth_visualization_path,crop_shape=crop_shape,s=args.s)
            #print('step 3 done')
            print('step3=',time.time()-time_step3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='final_select_pose_3d_stage',type=str)
    parser.add_argument('--Cam_list',type=list,default=[
        'CAMERA_01',
        'CAMERA_05',
        'CAMERA_06',
        'CAMERA_07',
        'CAMERA_08',
        'CAMERA_09'
    ])
    parser.add_argument('--train_val', default='train',type=str)
    parser.add_argument('--video_list', default=[0,150], nargs='+')
    parser.add_argument('--ex_str',default='extrinsics',choices=['pose','extrinsics'])
    parser.add_argument('--skip_step',default=1, type=int) #每skip_step个取一个record
    parser.add_argument('--sequence_len',default=200,type=int)
    parser.add_argument('--save_path',default='save/Stereo_Rectify/',type=str)
    parser.add_argument('--log',default='Stereo_Rectify/log.txt',type=str)
    parser.add_argument('--shift_json_path',type=str,default='') #Stereo_Rectify/shift_xy.json
    parser.add_argument('--datasets_path',type=str,
                        default='/root/autodl-tmp/datasets/DDAD/'
                        )
    parser.add_argument('--flip',type=bool,default=True)
    parser.add_argument('--crop',type=bool,default=True)
    parser.add_argument('--min_depth',type=float,default=0.00)
    parser.add_argument('--max_depth',type=float,default=150.)
    parser.add_argument('--mask_path',default='mask/DDAD/',type=str)
    parser.add_argument('--mask',default=True,type=bool)
    parser.add_argument('--rectify_shift_x',default= 3000,type=int)#3000 #在当前寻找crop的算法中，增加shift_x/y 会导致寻找overlap时间增加
    parser.add_argument('--rectify_shift_y',default= 1000,type=int)#1000
    parser.add_argument('--rectify_shift_x_stage2',default=0,type=int)
    parser.add_argument('--rectify_shift_y_stage2', default=0, type=int)
    parser.add_argument('--s', default=15, type=int)
    parser.add_argument('--s_stage2',default=2,type=int)
    parser.add_argument('--eps',default=0.00001,type=float)
    parser.add_argument('--json_file_path',default='datasets/DDAD/DDAD_video.json',type=str)
    parser.add_argument('--optimize_pose',default=True,type=bool)
    parser.add_argument('--specific_pose',default = '',type=str)#save/optimized_pose/DDAD/6Camera_8Frame/LoFTR/concat_final_pose/
    parser.add_argument('--BA_6cam_pose_s',type=str,
                        default='save/optimized_pose/DDAD/6Camera_8Frame/%s/')
    parser.add_argument('--distributed',default=False,type=bool)
    parser.add_argument('--processes_num',default=4,type=int)
    parser.add_argument('--specific_scene',default='',type=str) # Scene_0
    parser.add_argument('--select_pose_mode',default=False,type=bool)
    parser.add_argument('--specific_video_cam_JsonPath',default='20.json',type=str)
    #parser.add_argument('--gpu_num',type=str,default='0')
    args = parser.parse_args()

    if args.distributed:
        for train_val,video_list in zip(['train','val'],[[0,150],[150,200]]):

            args = parser.parse_args()
            args.train_val = train_val
            args.video_list = video_list
            print(args.video_list)
            processes = Pool(args.processes_num)
            for rank in range(args.processes_num):
                processes.apply_async(main_work,args=(rank,args))
            processes.close()
            processes.join()
        #os.system('shutdown')
        print('done')

    else:
        main_work(0,args)
