#-*- coding:utf-8 –*-

from keras.models import *
from keras.optimizers import *
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.layers import merge, Dropout, concatenate
from keras.layers.pooling import MaxPooling2D

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.utils.np_utils import to_categorical
import cv2 as cv
import glob
from keras.preprocessing.image import img_to_array

class Xnet(object):

    def __init__(self, patch_size, data_path, save_path, class_num):
        self.patch_size = patch_size
        self.data_path = data_path
        self.save_path = save_path
        self.class_num = class_num

    def unet_block(self, input):

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(256, 2, strides=(2, 2), padding='same')(drop5))
        # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(128, 2, strides=(2, 2), padding='same')(conv6))
        # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(64, 2, strides=(2, 2), padding='same')(conv7))
        # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Deconv2D(32, 2, strides=(2, 2), padding='same')(conv8))
        # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        #out = Conv2D(self.class_num, 1, activation='softmax')(conv9)

        return conv9

    def x_net(self):

        pre_input = Input((self.patch_size, self.patch_size, 3))
        cur_input = Input((self.patch_size, self.patch_size, 3))

        pre_conv = self.unet_block(pre_input)
        cur_conv = self.unet_block(cur_input)

        pre_output = Conv2D(self.class_num-1, 1, activation='softmax')(pre_conv)
        cur_output = Conv2D(self.class_num-1, 1, activation='softmax')(cur_conv)

        P_C = merge([pre_conv, cur_conv], mode='concat', concat_axis=3)

        change_cov = self.unet_block(P_C)

        change_output = Conv2D(2, 1, activation='softmax')(change_cov)

        model = Model(input=[pre_input, cur_input], outputs=[pre_output, cur_output, change_output])

        return model

    def generate_batch_data_random(self, path, batch_size):
        # 获取训练影像名称（不同时期影像、对应的label影像同名）
        imgs = glob.glob(path + "reference2016\\*.tif")
        np.random.shuffle(imgs)
        data_1 = []
        data_2 = []
        label_1 = []
        label_2 = []
        label_change = []
        cnt = 0
        while 1:
            for imgname in imgs:
                # 获取影像名称
                midname = imgname[imgname.rindex("\\") + 1:]

                # 读取影像
                img_2016 = cv.imread(path + "image2016\\" + midname)
                img_2017 = cv.imread(path + "image2017\\" + midname)
                gt_2016 = cv.imread(path + "reference2016\\" + midname, 0)
                gt_2017 = cv.imread(path + "reference2017\\" + midname, 0)
                # gt_changed = cv.imread(path + "reference_change\\" + midname, 0)

                # 影像灰度值归一化
                img_2016.astype('float32')
                img_2017.astype('float32')
                img_2016 /= 255
                img_2017 /= 255

                # reference影像由（row, col)转为（row，col，1）
                gt_2016 = img_to_array(gt_2016)
                gt_2017 = img_to_array(gt_2017)

                # 得到gt_change
                gt_changed = gt_2016 - gt_2017
                gt_changed[gt_changed != 0] = 1

                # 这台电脑的to_categorical有问题
                gt_2016 = to_categorical(gt_2016, self.class_num)[:, 1:self.class_num]
                gt_2017 = to_categorical(gt_2017, self.class_num)[:, 1:self.class_num]
                gt_changed = to_categorical(gt_changed, 2)

                gt_2016 = np.reshape(gt_2016, (1, self.patch_size, self.patch_size, self.class_num - 1))
                gt_2017 = np.reshape(gt_2017, (1, self.patch_size, self.patch_size, self.class_num - 1))
                gt_changed = np.reshape(gt_changed, (1, self.patch_size, self.patch_size, 2))

                data_1.append(img_2016)
                data_2.append(img_2017)
                label_1.append(gt_2016)
                label_2.append(gt_2017)
                label_change.append(gt_changed)

                cnt += 1
                if cnt == batch_size:
                    x = [np.array(data_1), np.array(data_2)]
                    y = [np.array(label_1), np.array(label_2), np.array(label_change)]
                    yield (x,y)
                    cnt = 0
                    data_1 = []
                    data_2 = []
                    label_1 = []
                    label_2 = []
                    label_change = []

    def train(self, net):

        model = net
        #model.load_weights('Xnet.hdf5')
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        callback_loss = ModelCheckpoint('Xnet_z2.hdf5', monitor='loss', verbose=1, save_best_only=True)

        callback_tensorboard = TensorBoard(log_dir=self.data_path, write_graph=True)
        batch_num = 2
        img_num = len(glob.glob(self.data_path + "reference2016\\*.tif"))
        model.fit_generator(self.generate_batch_data_random(self.data_path, batch_num),
                            samples_per_epoch=img_num, nb_epoch=300, callbacks=[callback_loss, callback_tensorboard],
                            verbose=1)


if __name__ == '__main__':

    patch_size = 512
    class_num = 16 #  算上背景
    #data_path = '/home/zhangchi/ChangeDetection/data/npy/'
    #save_path = '/home/zhangchi/ChangeDetection/data/ex1/'
    #data_path = 'E:\\DLspace\\ChangeDetection2\\source_data\\'
    #save_path = 'E:\\DLspace\\ChangeDetection2\\data\\result\\'
    data_path = '/media/zhangchi/文档/DLspace/ChangeDetection2/source_data/'
    save_path = '/media/zhangchi/文档/DLspace/ChangeDetection2/data/result/'
    Mynet = Xnet(patch_size, data_path, save_path, class_num)
    net = Mynet.x_net()
    Mynet.train(net)


        
        
























        









