import sys
sys.path.append('tools/RAFT_Stereo/core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from tools.RAFT_Stereo.core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'

def load_image(imfile,flip=False):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if flip:
        img = img[:,::-1,:].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():

        #right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        #left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = args.right_imgs
        left_images = args.left_imgs

        #print(f"Found {len(left_images)} images. Saving files to {output_directory}")
        #debug
        for (imfile1, imfile2) in list(zip(left_images, right_images)): #tqdm
            if args.flip:
                image1 = load_image(imfile2,args.flip)#flip需要左右视图反过来
                image2 = load_image(imfile1,args.flip)
            else:
                image1 = load_image(imfile1,args.flip)
                image2 = load_image(imfile2,args.flip)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            file_stem = imfile1.split('/')[-1]

            value = flow_up.cpu().numpy()#debug
            if args.save_numpy:
                np.save(output_directory / f"{file_stem[:-4]}.npy", flow_up.cpu().numpy().squeeze())
            #plt.imsave(output_directory / f"{file_stem}", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            #debug
            #print('max,min=',flow_up.max(),flow_up.min())
            #break

def RAFT_STEREO(l,r,output,flip=False):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='tools/RAFT_Stereo/models/raftstereo-middlebury.pth')
    parser.add_argument('--save_numpy', action='store_true', default=True, help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision',default=True)
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="alt",
                        help="correlation volume implementation",)
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    #below for run Stereo_Rectify by cmd
    parser.add_argument('--name')
    parser.add_argument('--Cam_list')
    parser.add_argument('--video_list', default=['%06d' % i for i in range(150, 160)], nargs='+')
    parser.add_argument('--ex_str')
    parser.add_argument('--skip_step')
    parser.add_argument('--sequence_len')
    parser.add_argument('--save_path')
    parser.add_argument('--log')
    parser.add_argument('--datasets_path')
    parser.add_argument('--rectify_shift_x')
    parser.add_argument('--rectify_shift_y')
    parser.add_argument('--s')
    parser.add_argument('--optimize_pose')
    parser.add_argument('--norm_T')
    parser.add_argument('--optimize_T')
    parser.add_argument('--optimize_pose_path')
    parser.add_argument('--specific_pose_path_s',  )
    parser.add_argument('--BA_6cam_pose_s')
    parser.add_argument('--gpu_num',)
    parser.add_argument('--flip',type=bool)
    args = parser.parse_args()

    args.left_imgs = l
    args.right_imgs = r
    args.output_directory = output
    args.flip = flip

    demo(args)

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true',default=True, help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
#--restore_ckpt models/raftstereo-middlebury.pth --corr_implementation alt --mixed_precision -l=../../tmp/StereoRectify/000189/CAMERA_01/l/*.png -r=../../tmp/StereoRectify/000189/CAMERA_01/r/*.png --save_numpy --output_directory ../../tmp/StereoRectify/000189/CAMERA_01/disp
    '''
