"""
数据增强
数据增强方法：
	直方图、水平翻转 垂直翻转 水平垂直翻转、高斯噪声、高斯模糊、旋转
"""

import os
import random
import cv2
import math
# Author :Fangyu_Zhou
# Coding :utf - 8 -*-
# Time : 2019/8/7 15:30
import cv2
import numpy as np
import glob
import os
class my_data_aug:

   def mean_filter(self, imgpath):
       img = cv2.imread(imgpath)
       blur = cv2.blur(img,(5,5))
       return blur

   def median_filter(self, imgpath):
       img = cv2.imread(imgpath)
       median = cv2.medianBlur(img,5)
       return median

   def gauss_filter(self, imgpath):
       img = cv2.imread(imgpath)
       gauss = cv2.GaussianBlur(img,(5,5),1)
       return gauss

   def bilateral_filter(self, imgpath):
       img = cv2.imread(imgpath)
       bilateral = cv2.bilateralFilter(img,7,50,50)
       return bilateral

   def erode_filter(self, imgpath):
       img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
       kernel = np.ones((3, 3), np.uint8)
       erosion = cv2.erode(img, kernel)
       return erosion


   def dilate_filter(self, imgpath):
       img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
       kernel = np.ones((3, 3), np.uint8)
       erosion = cv2.erode(img, kernel)
       dilation = cv2.dilate(erosion, kernel)
       return dilation



if __name__ == '__main__':
	img_dir = 'official_train_data'
	for root, dirs, files in os.walk(img_dir):
		for file in files:
			file_name = os.path.join(root, file)
			# print(file_name)


			imglist = glob.glob(file_name)
			#print(dirs)
			count = 0
			
			save_dir = 'additional_train_data' + root[19:] + '/'
			name = file.split('.jpg')[0]
			print(save_dir)
			# print(save_dir + name+ 'bilatera' + str(count) + '.jpg')


			for imgpath in imglist:

				ori_img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
				obj_my_data_aug = my_data_aug()
				blur = obj_my_data_aug.mean_filter(imgpath)
				median = obj_my_data_aug.median_filter(imgpath)
				gauss = obj_my_data_aug.gauss_filter(imgpath)
				shuangBian = obj_my_data_aug.bilateral_filter(imgpath)

				erosion = obj_my_data_aug.erode_filter(imgpath)
				dilation = obj_my_data_aug.dilate_filter(imgpath)

				blur_name = save_dir + name+ 'blur'+ str(count) + '.jpg'
				median_name = save_dir + name+ 'median' + str(count) + '.jpg'
				gauss_name = save_dir + name+ 'gauss' + str(count) + '.jpg'
				bilatera_name = save_dir + name+ 'bilatera' + str(count) + '.jpg'
				print(bilatera_name)

				ori_name = save_dir + name+ 'origin' + str(count) + '.jpg'
				erosion_name = save_dir + name+ 'after_erosion' + str(count) + '.jpg'
				dilate_name = save_dir + name+ 'after_dilate' + str(count) + '.jpg'

				cv2.imwrite(blur_name, blur)
				cv2.imwrite(median_name, median)
				cv2.imwrite(gauss_name, gauss)
				cv2.imwrite(bilatera_name, shuangBian)

				cv2.imwrite(ori_name, ori_img)
				cv2.imwrite(erosion_name, erosion)
				cv2.imwrite(dilate_name, dilation)

				count += 1


