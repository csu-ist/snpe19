#
# Copyright (c) 2016 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
import os

from PIL import Image

def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img_rgb = Image.open(img_filepath)
    img =img_rgb.convert("YCbCr")  # rgb to YUV

    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return img_ndarray

def __create_mean_raw(img_raw, mean_rgb):
    if img_raw.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_raw.shape[2])
    img_dim = (img_raw.shape[0], img_raw.shape[1])
    mean_raw_r = np.empty(img_dim)
    mean_raw_r.fill(mean_rgb[0])
    mean_raw_g = np.empty(img_dim)
    mean_raw_g.fill(mean_rgb[1])
    mean_raw_b = np.empty(img_dim)
    mean_raw_b.fill(mean_rgb[2])
    # create with c, h, w shape first
    tmp_transpose_dim = (img_raw.shape[2], img_raw.shape[0], img_raw.shape[1])
    mean_raw = np.empty(tmp_transpose_dim)
    mean_raw[0] = mean_raw_r
    mean_raw[1] = mean_raw_g
    mean_raw[2] = mean_raw_b
    # back to h, w, c
    mean_raw = np.transpose(mean_raw, (1, 2, 0))
    return mean_raw.astype(np.float32)


def __create_raw_incv3(img_filepath, mean_rgb, div, req_bgr_raw, save_uint8):
    img_raw = __get_img_raw(img_filepath)
#    mean_raw = __create_mean_raw(img_raw, mean_rgb)
    w, h, c = img_raw.shape
    print("yuv w, h, c:  ", w, h, c)
    snpe_raw = img_raw[:, :, 0] # get Y channel
    snpe_raw = snpe_raw.reshape([w, h, 1])
    snpe_raw = snpe_raw.astype(np.float32)
    # scalar data divide
    snpe_raw /= div  #255

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)

    return 0


#def __create_raw_incv3(img_filepath, mean_rgb, div, req_bgr_raw, save_uint8):
#    img_raw = __get_img_raw(img_filepath)
#    mean_raw = __create_mean_raw(img_raw, mean_rgb)
#    
#    snpe_raw = img_raw - mean_raw
#    snpe_raw = snpe_raw.astype(np.float32)
#    # scalar data divide
#    snpe_raw /= div
#
#    if req_bgr_raw:
#        snpe_raw = snpe_raw[..., ::-1]
#
#    if save_uint8:
#        snpe_raw = snpe_raw.astype(np.uint8)
#    else:
#        snpe_raw = snpe_raw.astype(np.float32)
#
#    img_filepath = os.path.abspath(img_filepath)
#    filename, ext = os.path.splitext(img_filepath)
#    snpe_raw_filename = filename
#    snpe_raw_filename += '.raw'
#    snpe_raw.tofile(snpe_raw_filename)
#
#    return 0

def __resize_square_to_jpg(src, dst, h, w):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2: src_img = src_img.convert(mode = 'RGB')
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    print("crop_coord: ", crop_coord)
    img = src_img.crop(crop_coord)
    # resize to alexnet size
    dst_img = img.resize((h, w), Image.ANTIALIAS)
    # save output - save determined from file extension
    dst_img.save(dst)
    return 0

def convert_img(src,dest,height, width):
    print("Converting images for inception v3 network.")

    print("Scaling to square: " + src)
    for root,dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
		if(True):
                    dest_image = os.path.join(dest, jpgs)
                    __resize_square_to_jpg(src_image,dest_image,height, width)

    print("Image mean: " + dest)
    for root,dirs,files in os.walk(dest):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
                mean_rgb=(128,128,128)
                __create_raw_incv3(src_image,mean_rgb,255.0,False,False)  #(src_image,mean_rgb,128,False,False)


def main():
    parser = argparse.ArgumentParser(description="Batch convert jpgs",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, required=True)
    parser.add_argument('-H','--Height',type=int, default=100)
    parser.add_argument('-W','--Width',type=int, default=100)
    parser.add_argument('-i','--img_folder',type=str, required=True)

    args = parser.parse_args()

    height = args.Height
    width = args.Width
    src = os.path.abspath(args.img_folder)
    dest = os.path.abspath(args.dest)

    convert_img(src,dest,height, width)

if __name__ == '__main__':
    exit(main())
