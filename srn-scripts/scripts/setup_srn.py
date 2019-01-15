#
# Copyright (c) 2017 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with SNPE SDK.
'''
import tensorflow as tf
import numpy as np
import os
import subprocess
import shutil
import hashlib
import argparse
import sys
import glob


INCEPTION_V3_PB_FILENAME            = 'srn_model_3L_16-32_0104.pb'
INCEPTION_V3_PB_OPT_FILENAME        = 'srn_model_3L_16-32_0104_opt.pb'
INCEPTION_V3_DLC_FILENAME           = 'srn_model_3L_16-32_0104_1.dlc'
INCEPTION_V3_DLC_QUANTIZED_FILENAME = 'srn_model_3L_16-32_0104_quantized.dlc'

INCEPTION_V3_LBL_FILENAME           = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'
RAW_LIST_FILE                       = 'raw_list.txt'
TARGET_RAW_LIST_FILE                = 'target_raw_list.txt'

def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)

def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()

def checkResource(inception_v3_data_dir, filename, md5):
    filepath = os.path.join(inception_v3_data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' + inception_v3_data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + ' does not match checksum of file ' + md5)

def find_optimize_for_inference():
    tensorflow_root = os.path.abspath(os.environ['TENSORFLOW_HOME'])
    for root, dirs, files in os.walk(tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)

def optimize_for_inference(model_dir, tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    pb_filename = ""

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
        pb_filename = INCEPTION_V3_PB_FILENAME
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        dlc_dir = os.path.join(model_dir, 'dlc')
        if not os.path.isdir(dlc_dir):
            os.makedirs(dlc_dir)
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(tensorflow_dir, INCEPTION_V3_PB_FILENAME),
               '--output', os.path.join(tensorflow_dir, INCEPTION_V3_PB_OPT_FILENAME),
               '--input_names', 'inputdata',  # import/
               '--output_names', 'g_net/dec1_0/BiasAdd'] 
        subprocess.call(cmd)
        pb_filename = INCEPTION_V3_PB_OPT_FILENAME

    return pb_filename

def prepare_data_images(snpe_root, model_dir, tensorflow_dir):
    # make a copy of the image files from the alexnet model data dir
    src_img_files = os.path.join(snpe_root, 'models', 'alexnet', 'data', '*.jpg')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/cropped')
    for file in glob.glob(src_img_files):
        shutil.copy(file, data_dir)

    # copy the labels file to the data directory
    # src_label_file = os.path.join(tensorflow_dir, INCEPTION_V3_LBL_FILENAME)
    # shutil.copy(src_label_file, data_dir)

    print('INFO: Creating SNPE inception_v3 raw data')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_raws_script = os.path.join(scripts_dir, 'create_srn_raws_2.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    cmd = ['python', create_raws_script,
           '-i', data_dir,
          # '-H',720,
          # '-W',1280,
           '-d',data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, TARGET_RAW_LIST_FILE),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)

def convert_to_dlc(pb_filename, model_dir, tensorflow_dir):
    print('INFO: Converting ' + pb_filename +' to SNPE DLC format')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    cmd = ['snpe-tensorflow-to-dlc',
           '--graph', os.path.join(tensorflow_dir, pb_filename),
           '--input_dim', 'inputdata', '1,256,256,1',
           '--out_node', 'g_net/dec1_0/BiasAdd',
           '--dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_FILENAME),
           '--allow_unconsumed_nodes']
    print(cmd)
    subprocess.call(cmd)

    print('INFO: Creating ' + INCEPTION_V3_DLC_QUANTIZED_FILENAME + ' quantized model')
    data_cropped_dir = os.path.join(os.path.join(model_dir, 'data'), 'cropped')
    cmd = ['snpe-dlc-quantize',
           '--input_dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_FILENAME),
           '--input_list', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '--output_dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_QUANTIZED_FILENAME)]
    print(cmd)
    subprocess.call(cmd)

def setup_assets(inception_v3_data_dir, download):

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')

    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)


    model_dir = os.path.join(snpe_root, 'models', 'srn')
    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    print('INFO: Extracting SRN TensorFlow model')
    tensorflow_dir = os.path.join(model_dir, 'tensorflow')
    if not os.path.isdir(tensorflow_dir):
        os.makedirs(tensorflow_dir)
#    cmd = ['tar', '-xzf',  os.path.join(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE), '-C', tensorflow_dir]
#    subprocess.call(cmd)

    pb_filename = optimize_for_inference(model_dir, tensorflow_dir)
    print("pb_filename:  ",pb_filename)

    prepare_data_images(snpe_root, model_dir, tensorflow_dir)

    convert_to_dlc(pb_filename, model_dir, tensorflow_dir)

    print('INFO: Setup inception_v3 completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the inception_v3 assets for tutorial examples.''')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                        help='directory containing the inception_v3 assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                        help='Download inception_v3 assets to inception_v3 example directory')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
