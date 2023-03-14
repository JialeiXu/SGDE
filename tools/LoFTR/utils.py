import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from scipy.spatial.transform import Rotation as scipy_R

def draw_LoFTR_matching(img1, img2, mkpts0, mkpts1, mconf, pdf_name='LoFTR-colab-demo.pdf'):

    img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # -----------  draw  ----------------------------------------------------------
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]

    make_matching_figure(img1,img2, mkpts0, mkpts1, color, mkpts0, mkpts1, text,
                         path=pdf_name)

def rotationMatrixToEulerAngles(R):
    r = scipy_R.from_matrix(R)
    res = r.as_rotvec()
    return res #for ceres
