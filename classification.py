#-*- coding:utf-8 –*-

from keras.models import *
import numpy as np
import gc

data_path = 'D:/张驰/ChangeDetection2/data/test/'
save_path = 'D:/张驰/ChangeDetection2/data/result/'
patch_size = 512
def max_ij(img):
    x = np.ndarray((patch_size, patch_size))
    y = np.ndarray((patch_size, patch_size))
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            a = img[i][j]
            b = np.argmax(a, axis=0)
            x[i, j] = b + 1
            y[i, j] = a.max()
    return x, y

def get_result():

    model = load_model('Xnet_i.hdf5')
    xy = np.loadtxt(data_path + "test_xy.txt", dtype=np.int)
    num_each_prediction = 100
    cnt = int(len(xy)/num_each_prediction) + 1
    for n in range(cnt):
        data_pre_val = np.load(data_path + "data_1_val.npy")
        data_cur_val = np.load(data_path + "data_2_val.npy")

        a = n * num_each_prediction
        b = min((n + 1) * num_each_prediction, len(xy))
        data_pre_val = data_pre_val[a:b]
        data_cur_val = data_cur_val[a:b]

        data_pre_val.astype('float32')
        data_cur_val.astype('float32')
        data_pre_val = data_pre_val / 255
        data_cur_val = data_cur_val / 255

        predict_pre, predict_cur, predict_change = model.predict([data_pre_val, data_cur_val], batch_size=1, verbose=1)
        del predict_pre, data_pre_val, data_cur_val
        gc.collect()
        label_normal_2017 = np.ndarray(((b-a), patch_size, patch_size), dtype=np.uint8) # label
        probability_2017 = np.ndarray(((b-a), patch_size, patch_size), dtype=np.float16) #label对应的概率
        label_normal_change = np.ndarray(((b - a), patch_size, patch_size), dtype=np.uint8)  # label
        probability_change = np.ndarray(((b - a), patch_size, patch_size), dtype=np.float16)  # label对应的概率

        for i in range(0, (b-a)):
            label_normal_2017[i], probability_2017[i] = max_ij(predict_cur[i])
            label_normal_change[i], probability_change[i] = max_ij(predict_change[i])

        np.save(save_path + "predict_2017" + str(n+1) + ".npy", label_normal_2017)
        np.save(save_path + "predict_change" + str(n + 1) + ".npy", label_normal_change)
        np.save(save_path + "predict_2017_p" + str(n + 1) + ".npy", probability_2017)
        np.save(save_path + "predict_change_p" + str(n + 1) + ".npy", probability_change)
        del predict_cur, predict_change, data_pre_val, data_cur_val
        del label_normal_2017, label_normal_change, probability_2017, probability_change
        gc.collect()


if __name__ == '__main__':

    get_result()












