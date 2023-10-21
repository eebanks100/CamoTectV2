import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

import time
import cv2
import shutil
from scipy import misc
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./weights/SINet_V2/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['Dataset']: 
    outputSave = time.strftime("%m-%d-%Y, %H-%M-%S")
    resPath = './sinet_res/{}/'.format(outputSave)
    resCopy = './sinet_output/'

    model = Network(imagenet_pretrained=False)

# Elisha - use GPU if available, if not then use CPU
    if torch.cuda.is_available():     
        device = torch.device('cuda')
        model.cuda()
        model.load_state_dict(torch.load(opt.pth_path, map_location=device))
    else:
         device = torch.device('cpu')
         model.load_state_dict(torch.load(opt.pth_path, map_location=device))
    
    #model.load_state_dict(torch.load(opt.pth_path))
    model.eval()

    os.makedirs(resPath, exist_ok=True)
    image_root = 'Dataset/Test/Imgs/'

# Elisha - running SINetV2 test will no longer use ground truth imgs for test imgs
    gt_root = None
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, name, original_size, _ = test_loader.load_data()
        # gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)

# Elisha - use GPU if available, if not then use CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            image = image.to(device)
        else:
            device = torch.device('cpu')
            image = image.to(device)

        h, w = original_size
        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=(h, w), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(resPath+name,res*255)

print('SINet results saved at: ' + os.path.join(resPath, outputSave))

# Elisha - copy the most recent SINetV2 test run for sinetProc to use 
if not os.path.exists(resCopy):
    shutil.copytree(resPath, resCopy)
else:
    shutil.rmtree(resCopy, ignore_errors=True)
    shutil.copytree(resPath, resCopy)