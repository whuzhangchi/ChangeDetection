#-*- coding:utf-8 –*-

import numpy as np
import cv2 as cv
import gc

source_data_path = 'E:\\DLspace\\ChangeDetection2\\source_data\\'
data_path = 'E:\\DLspace\\ChangeDetection2\\data\\result1\\'
save_path = 'E:\\DLspace\\ChangeDetection2\\result\\'

w = 256
patch_size = 512

def result():
    test_xy = np.loadtxt(source_data_path + "test_xy.txt", dtype=np.int)
    num_each_prediction = 100
    cnt = int(len(test_xy) / num_each_prediction) + 1
    row = 11827
    col = 17833
    result_2016 = np.zeros((row, col))
    result_2017 = np.zeros((row, col))
    result_change = np.zeros((row, col))
    '''
    result_2016_p = np.zeros((row, col))
    result_2017_p = np.zeros((row, col))
    result_change_p = np.zeros((row, col))
    '''


    for n in range(cnt):
        pre_2016 = np.load(data_path + "predict_2016_" + str(n+1) + ".npy") # 2016年影像分类结果
        pre_2017 = np.load(data_path + "predict_2017_" + str(n + 1) + ".npy")  # 2017年影像分类结果
        pre_change = np.load(data_path + "predict_change_" + str(n+1) + ".npy")  # 变化结果
        '''
        pre_2016_p = np.load(data_path + "predict_2017_p" + str(n + 1) + ".npy")  # 2017年影像分类结果label对应的概率
        pre_2017_p = np.load(data_path + "predict_2017_p" + str(n+1) + ".npy")  # 2017年影像分类结果label对应的概率
        pre_change_p = np.load(data_path + "predict_change_p" + str(n+1) + ".npy")  # 变化结果label对应的概率
        '''
        a = n * num_each_prediction
        b = min((n + 1) * num_each_prediction, len(test_xy))
        xy = test_xy[a:b]
        for i in range(len(xy)):
            x = xy[i, 0]
            y = xy[i, 1]
            result_2016[x - w:x + w, y - w:y + w] = pre_2016[i]
            result_2017[x - w:x + w, y - w:y + w] = pre_2017[i]
            result_change[x - w:x + w, y - w:y + w] = pre_change[i]
            '''
            result_2016_p[x - w:x + w, y - w:y + w] = pre_2016_p[i] * 100
            result_2017_p[x - w:x + w, y - w:y + w] = pre_2017_p[i] * 100
            result_change_p[x - w:x + w, y - w:y + w] = pre_change_p[i] * 100
            del pre_change_p, pre_change, pre_2017_p, pre_2017
            '''
    cv.imwrite(save_path + 'result_2016.tif', result_2016)
    cv.imwrite(save_path + 'result_2017.tif', result_2017)
    cv.imwrite(save_path + 'result_change.tif', result_change)

    #cv.imwrite(save_path + 'result_2017_p.tif', result_2017_p)
    #cv.imwrite(save_path + 'result_change_p.tif', result_change_p)
    '''
    np.save(save_path + 'result_2017.npy', result_2017)
    np.save(save_path + 'result_2017_p.npy', result_2017_p)
    np.save(save_path + 'result_change.npy', result_change)
    np.save(save_path + 'result_change_p.npy', result_change_p)
    print(result_change_p.min(), result_change_p.max())
    '''


if __name__ == '__main__':

    result()








