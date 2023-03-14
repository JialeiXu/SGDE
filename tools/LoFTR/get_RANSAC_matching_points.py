import multiprocessing
from multiprocessing import Pool
import math, os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.cm as cm
import cv2
import glob
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
import argparse
import json
import shutil
from scipy.spatial.transform import Rotation as scipy_R


def init_file(path):
    # if os.path.exists(path):
    # shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/LoFTR_result', exist_ok=True)
    os.makedirs(path + '/Matching_Points', exist_ok=True)


def rotationMatrixToEulerAngles(R):
    r = scipy_R.from_matrix(R)
    res = r.as_rotvec()
    return res  # for ceres


def fix_focal_length(img1, intrinsics1, intrinsics2):
    # fix img2, resize img1
    shape = img1.shape
    fx1, fy1 = intrinsics1[0, 0].copy(), intrinsics1[1, 1].copy()
    fx2, fy2 = intrinsics2[0, 0].copy(), intrinsics2[1, 1].copy()
    img1 = cv2.resize(img1, (int(shape[1] * fy2 / fy1), int(shape[0] * fx2 / fx1)))
    intrinsics1[0] = intrinsics1[0] * fx2 / fx1
    intrinsics1[1] = intrinsics1[1] * fy2 / fy1
    return img1, intrinsics1


def fix_focal_length_cam01(img, f_before, f_after):
    shape = img.shape
    img1 = cv2.resize(img, (int(shape[1] * f_after / f_before),
                            int(shape[0] * f_after / f_before)))

    return img1


def epipolarConstrain(K1, K2, R, T, kpts0, kpts1, mkpts1):
    T = np.expand_dims(T, axis=1)
    kpts0 = np.transpose(kpts0, (1, 0))
    one = np.ones_like(kpts0[0:1])
    kpts0_3 = np.vstack([kpts0, one])

    kpts1_3_gt = R @ kpts0_3 + T
    kpts1_3_gt = np.transpose(kpts1_3_gt, (1, 0))

    error_x, error_y = np.abs(kpts1_3_gt[:, 0] - kpts1[:, 0]), np.abs(kpts1_3_gt[:, 1] - kpts1[:, 1])
    print(error_x.max(), error_x.min(), error_y.max(), error_y.min())

    return None


def draw_lines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        r = r[0]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def Epilines(K1, K2, R, T, mkpts0, mkpts1):
    fundamental_matrix, mask = cv2.findFundamentalMat(mkpts0, mkpts1)
    S = np.mat([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = S @ R
    F_2 = (np.linalg.pinv(K2).T) @ E @ np.linalg.pinv(K1)

    if True:  # use GT from datasets
        F = F_2
    else:
        F = fundamental_matrix
    lines0 = cv2.computeCorrespondEpilines(mkpts1, 2, F)
    lines1 = cv2.computeCorrespondEpilines(mkpts0, 1, F)
    if False:  # draw Epilines
        # lines1 = cv2.computeCorrespondEpilines(mkpts1,2,F_2)
        img1_resize_padding_gray = cv2.cvtColor(img1_resize_padding.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_raw_gray = cv2.cvtColor(img2_raw.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img_draw_1, img_draw_2 = draw_lines(img1_resize_padding_gray, img2_raw_gray, lines0, mkpts0, mkpts1)
        img_draw_4, img_draw_3 = draw_lines(img2_raw_gray, img1_resize_padding_gray, lines1, mkpts1, mkpts0)

        cv2.imwrite('tmp/1.png', img_draw_1)
        cv2.imwrite('tmp/2.png', img_draw_2)
        cv2.imwrite('tmp/3.png', img_draw_3)
        cv2.imwrite('tmp/4.png', img_draw_4)

    return lines0, lines1


def findEssentialMat(kpts0, kpts1):
    """
    Find Essential Matrix from points in the camera coordinates
    Args:
        kpts0: (n, 2) - <x, y>
        kpts1: (n, 2) - <x, y>

    Returns:
        E: (3, 3) - essential matrix
    """
    # np 2 tensor
    kpts0, kpts1 = torch.from_numpy(kpts0), torch.from_numpy(kpts1)

    xx = torch.cat([kpts0, kpts1], dim=1).transpose(0, 1)  # (4, n)
    X = torch.stack([
        xx[2] * xx[0], xx[2] * xx[1], xx[2],
        xx[3] * xx[0], xx[3] * xx[1], xx[3],
        xx[0], xx[1], torch.ones_like(xx[0])
    ], dim=0)  # (9, n)
    XX = torch.matmul(X, X.transpose(0, 1))  # (9, 9)
    e, v = torch.linalg.eigh(XX, UPLO='U')
    e_hat = v[:, 0]
    e_hat = e_hat / torch.norm(e_hat)
    E = e_hat.reshape((3, 3)).numpy().astype(np.float64)
    return E


def LoFTR_matching(img1, img2):
    image_type = 'outdoor'
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()

    img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    img1_tensor = torch.from_numpy(img1)[None][None].cuda() / 255.
    img2_tensor = torch.from_numpy(img2)[None][None].cuda() / 255.
    batch = {'image0': img1_tensor, 'image1': img2_tensor}

    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # -----------  draw  ----------------------------------------------------------
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    # fig = make_matching_figure(img1_raw, img2_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)

    # A high-res PDF will also be downloaded automatically.
    if False:  # show match result #draw LoFTR 结果已经集成到fun:draw_LoFTR_matching中
        make_matching_figure(img1, img2, mkpts0, mkpts1, color, mkpts0, mkpts1, text,
                             path="LoFTR-colab-demo.pdf")
    return mkpts0, mkpts1, mconf


def draw_LoFTR_matching(img1, img2, mkpts0, mkpts1, mconf, pdf_name='LoFTR-colab-demo.pdf'):
    img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # -----------  draw  ----------------------------------------------------------
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]

    make_matching_figure(img1, img2, mkpts0, mkpts1, color, mkpts0, mkpts1, text,
                         path=pdf_name)


def extract_global_ex(args, cam_name, item, Pose_cam2cam_01):
    cam_01 = np.load(args.datasets_path + str(item['timestamp']) + '_' + 'CAMERA_01' + '.npz')
    pose = cam_01['pose'] @ Pose_cam2cam_01[cam_name]
    return pose


def extract_cam_pose2cam_01(args, dict):
    pose2cam_01 = {}
    camera_list = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06',
                   'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
    cam_01 = np.load(args.datasets_path + str(dict[0]['timestamp']) + '_' + 'CAMERA_01' + '.npz')
    ex_01 = cam_01['extrinsics']
    for cam_name in camera_list:
        cam_else = np.load(args.datasets_path + str(dict[0]['timestamp']) + '_' + cam_name + '.npz')
        ex_else = cam_else['extrinsics']
        ex = np.linalg.pinv(ex_01) @ ex_else
        pose2cam_01[cam_name] = ex
    pose2cam_01['CAMERA_01'] = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    return pose2cam_01


def save_K_ex(args, Pose_cam2cam_01, dict, save_path):
    os.makedirs(save_path, exist_ok=True)
    # save cam 1 to cam i
    with open(save_path + 'Cam_1ToCam_i.txt', 'a') as f:
        for k, v in Pose_cam2cam_01.items():
            v = np.linalg.pinv(v)
            po3 = rotationMatrixToEulerAngles(v[:3, :3])
            f.write(k + ' ' + str(po3[0]) + ' ' + str(po3[1]) + ' ' + str(po3[2]) + ' ')
            f.write(str(v[0, 3]) + ' ' + str(v[1, 3]) + ' ' + str(v[2, 3]) + '\n')
    # save K for all cameras
    cam_name_list = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
    with open(save_path + 'intrinsics.txt', 'a') as f:
        for cam_name_i in cam_name_list:
            cam = np.load(args.datasets_path + str(dict[0]['timestamp']) + '_' + cam_name_i + '.npz')
            K = cam['intrinsics']
            if cam_name_i == 'CAMERA_01':
                K[0] = K[0] / K[0, 0] * args.resize_f
                K[1] = K[1] / K[1, 1] * args.resize_f
            f.write(
                cam_name_i + ' ' + str(K[0, 0]) + ' ' + str(K[1, 1]) + ' ' + str(K[0, 2]) + ' ' + str(K[1, 2]) + '\n')

    # save world cor to cam cor for cam_01
    with open(save_path + 'cam01_pose.txt', 'a') as f:
        cam_name = 'CAMERA_01'
        for dict_i in dict:
            cam = np.load(args.datasets_path + str(dict_i['timestamp']) + '_' + cam_name + '.npz')
            pose = np.linalg.pinv(cam['pose'])
            po3 = rotationMatrixToEulerAngles(pose[:3, :3])
            f.write(str(dict_i['timestamp']) + ' ' + str(po3[0]) + ' ' + str(po3[1]) + ' ' + str(po3[2]) + ' ')
            f.write(str(pose[0, 3]) + ' ' + str(pose[1, 3]) + ' ' + str(pose[2, 3]) + '\n')

def main_work(rank,args):
    args.rank=rank

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    args.datasets_path += args.train_val + '/'
    args.video_list = ['%06d' % i for i in range(int(args.video_list[0]), int(args.video_list[1]))]
    json_file = json.load(open(args.json_file_path))
    #for video_i in tqdm(range(len(args.video_list))):
    for video_i in range(args.rank,len(args.video_list),args.processes_num):
        skip_num = -1
        args.total_mkpts_num = 0
        init_file(path=args.save_path + str(args.video_list[video_i]))
        dict_all = []
        for i in range(len(json_file[args.train_val])):
            if json_file[args.train_val][i]['video_num'] == args.video_list[video_i] \
                    and json_file[args.train_val][i]['Camera'] == 'CAMERA_01':  # cam_01只是随意指定，避免6倍重复数据
                skip_num += 1
                if skip_num < args.skip_step:
                    dict_all.append(json_file[args.train_val][i])
        Pose_cam2cam_01 = extract_cam_pose2cam_01(args, dict_all)
        save_K_ex(args, Pose_cam2cam_01, dict_all, save_path=args.save_path + str(args.video_list[video_i]) + '/')
        ### save world cor 2 cam cor for cam1
        ### save K for all camera after resize
        ### save cam1 to cam i

        for Cam_l_i in range(len(args.Cam_list_l)):

            # for Cam_r_i in range(Cam_l_i, len(args.Cam_list_r)):
            cam_l = args.Cam_list_l[Cam_l_i]  # str
            cam_r = args.Cam_list_r[Cam_l_i]  # str  #Cam_r_i

            for dict_l in range(len(dict_all)):
                for dict_r in range(len(dict_all)):
                    if cam_l == cam_r and dict_l == dict_r:  # 相同相机，相同时间戳, continue
                        continue
                    # step 1, 确定匹配图片对
                    item_l = dict_all[dict_l]
                    item_r = dict_all[dict_r]
                    # load mask
                    if args.mask:
                        mask_l_path = args.mask_path + str(cam_l) + '_' + str(item_l['scene'] + '.png')
                        mask_r_path = args.mask_path + str(cam_r) + '_' + str(item_r['scene'] + '.png')
                        mask_l = cv2.imread(mask_l_path)[:, :, 0]
                        mask_r = cv2.imread(mask_r_path)[:, :, 0]
                        mask_l = cv2.resize(mask_l, ())
                        mask_r = cv2.resize(mask_r, )
                    # --------
                    cam_L = np.load(args.datasets_path + str(item_l['timestamp']) + '_' + cam_l + '.npz')
                    cam_R = np.load(args.datasets_path + str(item_r['timestamp']) + '_' + cam_r + '.npz')
                    img1_raw = cv2.cvtColor(cam_L['rgb'].astype(np.uint8), cv2.COLOR_RGB2BGR)
                    img2_raw = cv2.cvtColor(cam_R['rgb'].astype(np.uint8), cv2.COLOR_RGB2BGR)
                    K_l, K_r = cam_L['intrinsics'], cam_R['intrinsics']
                    pose_l = extract_global_ex(args,cam_l, item_l, Pose_cam2cam_01)
                    pose_r = extract_global_ex(args,cam_r, item_r, Pose_cam2cam_01)
                    if cam_l == 'CAMERA_01' and cam_r != 'CAMERA_01':
                        K_l[0] = K_l[0] / K_l[0, 0] * 1060
                        K_l[1] = K_l[1] / K_l[1, 1] * 1060

                    if cam_l == 'CAMERA_01' and cam_r != 'CAMERA_01':
                        f_after = 1060.  # focal length of cam_01
                        img1_resize = fix_focal_length_cam01(img1_raw, cam_L['intrinsics'][0][0], f_after)

                        img1_zero = np.zeros_like(img1_raw)
                        resize_shape = img1_resize.shape
                        img1_zero[:resize_shape[0], :resize_shape[1]] = img1_resize
                        img_l, img_r = img1_zero, img2_raw
                    else:
                        img_l, img_r = img1_raw, img2_raw

                    # step 3, Matching Points with LoFTR
                    matching_points_save_path = args.raw_path + str(
                        args.video_list[video_i]) + '/Matching_Points/' + \
                                                cam_l + 'T' + cam_r + str(item_l['timestamp']) + 'T' + str(
                        item_r['timestamp']) + '.npz'
                    if args.save_point_or_Ransac[1]:  # load matching points
                        matching_points = np.load(matching_points_save_path)
                        mkpts0, mkpts1, mconf = matching_points['mkpts0'], matching_points['mkpts1'], matching_points[
                            'mconf']

                    # step 4, robust fitting
                    # -----------------
                    # RANSAC
                    if args.RANSAC_method == 'RANSAC':
                        RANSAC_method = cv2.RANSAC
                    elif args.RANSAC_method == 'USAC_ACCURATE':
                        RANSAC_method = cv2.USAC_ACCURATE
                    elif args.RANSAC_method == 'cv.USAC_MAGSAC':
                        RANSAC_method = cv2.USAC_MAGSAC

                    kpts0 = (mkpts0 - K_l[[0, 1], [2, 2]][None]) / K_l[[0, 1], [0, 1]][None]
                    kpts1 = (mkpts1 - K_r[[0, 1], [2, 2]][None]) / K_r[[0, 1], [0, 1]][None]
                    ransac_thr = 1.0 / np.mean([K_l[0, 0], K_r[1, 1], K_l[0, 0], K_r[1, 1]]) / args.RANSAC_div
                    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=0.99999,
                                                   method=RANSAC_method)
                    mask = mask.ravel() > 0
                    mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]
                    mconf = mconf[mask]

                    # ----------------
                    if True:

                        ex = np.linalg.pinv(pose_r) @ pose_l
                        R, T = ex[:3, :3], ex[:3, 3]
                        line0, line1 = Epilines(K_l, K_r, R, T, mkpts0, mkpts1)

                        error_piexl_num = np.zeros_like(mkpts0[:, 0])
                        for point_i in range(mkpts0.shape[0]):
                            x_i = mkpts1[point_i][0]
                            y_i = mkpts1[point_i][1]
                            y_line = -(line1[point_i][0][0] * x_i + line1[point_i][0][2]) / line1[point_i][0][1]
                            error_piexl_num[point_i] = np.abs(y_i - y_line)
                        mask = error_piexl_num < args.error_pixel
                        mkpts0 = mkpts0[mask]
                        mkpts1 = mkpts1[mask]
                        mconf = mconf[mask]
                    if False:  # show LoFTR result
                        pdf_path = args.save_path + str(
                            args.video_list[video_i]) + '/LoFTR_result/' + cam_l + 'T' + cam_r + str(
                            item_l['timestamp']) + 'T' + str(item_r['timestamp']) + '.pdf'
                        draw_LoFTR_matching(img_l, img_r, mkpts0, mkpts1, mconf, pdf_path)

                    # step 5,  triangulate

                    mkpts0, mkpts1 = np.swapaxes(mkpts0, 1, 0), np.swapaxes(mkpts1, 1, 0)
                    mkpts0_copy, mkpts1_copy = mkpts0.copy(), mkpts1.copy()

                    mkpts0[0] = (mkpts0[0] - K_l[0, 2]) / K_l[0, 0]
                    mkpts0[1] = (mkpts0[1] - K_l[1, 2]) / K_l[1, 1]
                    mkpts1[0] = (mkpts1[0] - K_r[0, 2]) / K_r[0, 0]
                    mkpts1[1] = (mkpts1[1] - K_r[1, 2]) / K_r[1, 1]

                    if mkpts0.shape[1] == 0: continue
                    points4d = cv2.triangulatePoints(
                        np.linalg.pinv(pose_l)[:3],
                        np.linalg.pinv(pose_r)[:3],
                        mkpts0, mkpts1)

                    for i in range(3):
                        points4d[i] = points4d[i] / points4d[3]

                    # step 6: save txt
                    with open(args.save_path + str(args.video_list[video_i]) + '/mkpts.txt', 'a') as f:
                        for i in range(mkpts0.shape[1]):
                            f.write(str(mkpts0_copy[0, i]) + ' ' + str(mkpts0_copy[1, i]) + '\n')
                            f.write(str(mkpts1_copy[0, i]) + ' ' + str(mkpts1_copy[1, i]) + '\n')
                    with open(args.save_path + str(args.video_list[video_i]) + '/points.txt', 'a') as f:
                        for i in range(points4d.shape[1]):
                            f.write(str(points4d[0][i]) + ' ' + str(points4d[1][i]) + ' ' + str(points4d[2][i]) + '\n')

                    with open(args.save_path + str(args.video_list[video_i]) + '/cameras_index.txt', 'a') as f:
                        for i in range(mkpts0.shape[1]):
                            f.write(str(item_l['timestamp']) + ' ' + cam_l + '\n')
                            f.write(str(item_r['timestamp']) + ' ' + cam_r + '\n')
                    args.total_mkpts_num += mkpts0.shape[1]
        with open(args.save_path + str(args.video_list[video_i]) + '/num.txt', 'a') as f:
            f.write('6' + ' ')  # relative cam, number of the other cameras
            f.write(str(len(dict_all)) + ' ')  # num_timestamp, need to be edit
            f.write(str(args.total_mkpts_num) + ' ')  # points
            f.write(str(args.total_mkpts_num * 2) + ' ')  # Num_observations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Cam_list_l', type=list, default=[
        'CAMERA_01',
        'CAMERA_06',
        'CAMERA_08',
        'CAMERA_09',
        'CAMERA_07',
        'CAMERA_01',
    ])
    parser.add_argument('--Cam_list_r', type=list, default=[
        'CAMERA_06',
        'CAMERA_08',
        'CAMERA_09',
        'CAMERA_07',
        'CAMERA_05',
        'CAMERA_05',
    ])

    parser.add_argument('--video_list', default=[0,200], nargs='+')
    parser.add_argument('--train_val', default='train', type=str)
    parser.add_argument('--ex_str', default='extrinsics', choices=['pose', 'extrinsics'])
    parser.add_argument('--raw_path', default='save_matching_points/DDAD/6Camera_8Frame/raw_matching_points/')
    parser.add_argument('--save_path', default='save_matching_points/DDAD/6Camera_8Frame/USAC_ACCURATE/', type=str)
    parser.add_argument('--log', default='Stereo_Rectify/log.txt', type=str)
    parser.add_argument('--datasets_path', type=str,
                        default=''
                        )
    parser.add_argument('--skip_step', type=int, default=8)
    parser.add_argument('--json_file_path', default='../../datasets/DDAD/DDAD_video.json', type=str)
    parser.add_argument('--kpts', type=list, default=[None, None])
    parser.add_argument('--error_pixel', default=50., type=float)
    parser.add_argument('--mask_path', default='../../mask/DDAD/', type=str)
    parser.add_argument('--mask', default=False, type=bool)
    # parser.add_argument('--save_pose_path',type=str,default='../../tmp/optimize_pose/')
    # parser.add_argument('--match_points_path', type=str, default='../../tmp/optimize_pose/match_points/50pixel_div10/')
    parser.add_argument('--resize_f', type=float, default=1060.)
    parser.add_argument('--RANSAC_div', default=1., type=float)
    parser.add_argument('--total_mkpts_num', default=0, type=int)
    parser.add_argument('--RANSAC_method', default='USAC_ACCURATE', type=str)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--save_point_or_Ransac', default=[False, True], type=list)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--processes_num', default=30, type=int)

    args = parser.parse_args()

    if args.distributed:
        processes = Pool(args.processes_num)
        for rank in range(args.processes_num):
            processes.apply_async(main_work,args=(rank,args))
        processes.close()
        processes.join()
    else:
        main_work(0,args)
    print('done')
