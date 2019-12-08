# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2019/12/06
#   description:
#
#================================================================

import json

if __name__ == "__main__":

    f = open('./A3D_2.0.json')
    dic = json.load(f)

    min_num_frames = 1000
    min_normal = 1000
    max_frames = 0
    for k in dic.keys():
        min_num_frames = min(min_num_frames, dic[k]['num_frames'])
        max_frames = max(max_frames, dic[k]['num_frames'])
        if(dic[k]['num_frames'] > 300):
            print(k) 
        if (dic[k]['anomaly_start'] == None or dic[k]['anomaly_end'] == None):
            # print(k)
            continue
        if dic[k]['anomaly_start'] < 16:
            # print(k)
            continue
        min_normal = min(min_normal, dic[k]['anomaly_start'])
    print('min frames: ', min_num_frames)
    print('min normal frames: ', min_normal)
    print('max_frames', max_frames)
