# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:34:23 2018

@author: wywei
"""

import numpy as np
import os
import scipy.misc

from PIL import Image

def __get_img_raw(img_filepath, out_y_path):
    img_filepath = os.path.abspath(img_filepath)
    img_rgb = Image.open(img_filepath)
    img =img_rgb.convert("YCbCr")  # rgb to YUV

    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    out_y = np.fromfile(out_y_path, np.float32)  
    if len(out_y.shape) != 2: # if shape is w, h, c
        out_y = out_y.reshape([700, 700])
    print(out_y)
        
    t = np.clip(out_y, 0.0, 1.0) * 255.0
    t = t.astype(np.uint8)
    img_ndarray[:, :, 0] = t   # replace y'(0~255) to orignal y 
    print(t)
    img_ndarray = Image.fromarray(img_ndarray, mode="YCbCr")
    print(img_ndarray.mode)
    img_ndarray = img_ndarray.convert("RGB")
    scipy.misc.imsave(out_y_path.split('.')[0]+'-srn-charis_700x700_out-0-complex_outy.png', img_ndarray)
    print(out_y_path.split('.')[0]+'-srn-charis_700x700_out.png')
    return img_ndarray

#/opt/SNPE/snpe-1.19.2/models/srn/output/Result_0/g_net/dec1_0_2/BiasAdd:0.raw
#/opt/SNPE/snpe-1.19.2/models/srn/data/cropped/chairs.jpg

__get_img_raw("/opt/SNPE/snpe-1.19.2/models/srn/data/cropped_700x700/chairs.jpg", "/opt/SNPE/snpe-1.19.2/models/srn/data/cropped_700x700/srn_out_chairs_700x700_20.raw")
