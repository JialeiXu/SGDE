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
    if False: #show match result #draw LoFTR 结果已经集成到fun:draw_LoFTR_matching中
        make_matching_figure(img1,img2, mkpts0, mkpts1, color, mkpts0, mkpts1, text,
                             path="LoFTR-colab-demo.pdf")
    return mkpts0,mkpts1,mconf

def draw_LoFTR_matching(img1, img2, mkpts0, mkpts1, mconf, pdf_name='LoFTR-colab-demo.pdf'):

    #img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # -----------  draw  ----------------------------------------------------------
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    text=[]

    make_matching_figure(img1,img2, mkpts0, mkpts1, color, mkpts0, mkpts1, text,
                         path=pdf_name)


if __name__=='__main__':
    root_path = '/root/autodl-tmp/datasets/DDAD/my_DDAD/'
    img_l = np.load(root_path + '15609721105309222_CAMERA_05.npz')['rgb']
    img_r = np.load(root_path + '15609721105309222_CAMERA_01.npz')['rgb']
    print(img_r.shape)
    img_l = cv2.resize(img_l,(1936//2, 1216//2))
    img_r = cv2.resize(img_r,(1936//2, 1216//2))
    print(img_r.shape)


    torch.cuda.empty_cache()
    mkpts0, mkpts1, mconf = LoFTR_matching(img_l, img_r)
    print(mkpts0.shape)
    for i in range(mkpts0.shape[0]):
        if mkpts0[i][0]<600 or mkpts1[i][0]>300:
            mkpts0[i] = mkpts0[0]
            mkpts1[i] = mkpts1[0]

    pdf_path = '/root/autodl-tmp/Project/FSDE/tmp/Loftr/' + '1.png'
    draw_LoFTR_matching(img_l,img_r,mkpts0,mkpts1,mconf,pdf_path)
