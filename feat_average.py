# -*- coding: utf-8 -*-
#================================================================
#   God Bless You. 
#   
#   author: xiziwang 
#   email: xiziwang@iu.edum
#   created date: 2019/11/20
#   description: 
#
#================================================================

import os
import numpy as np

if __name__ == "__main__":

    root = '/home/data/vision7/A3D_frame_feat/val/'
    outdir = '/home/data/vision7/A3D_feat/val/'
    
    # load numpy arrays
    for video_dir in os.listdir(root):
        video_path = os.path.join(root, video_dir)
        data = list() 
        for npfile in os.listdir(video_path):
            data.append(np.load(os.path.join(video_path, npfile) ) )
        data = np.vstack(data)
        np.save(os.path.join(outdir, video_dir+'.npy'), np.average(data, axis = 0)) 
