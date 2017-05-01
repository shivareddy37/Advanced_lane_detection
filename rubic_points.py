import cv2
import numpy as np
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#testing for camera claibration
# cam_mat, dist_coff =  utils.camera_calibration()
# img = cv2.imread('test_images/test3.jpg',1)
# undistorted_im = cv2.undistort(img,cam_mat, dist_coff, None, cam_mat)
# # cv2.imshow('original image', img)
# # cv2.imshow('undistorted image', undistorted_im)
# # cv2.waitKey(0)
# cv2.imwrite('projectstages_out_images/test3.png',img)
# cv2.imwrite('projectstages_out_images/test3_undistorted.png', undistorted_im)


# testing prespective transform
# img = cv2.imread('test_images/test3.jpg',1)
# mat = utils.get_prespective(img)
# # cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
# # cv2.namedWindow('wrapped image', cv2.WINDOW_NORMAL)
# #
# # cv2.imshow('original image', img)
# # cv2.imshow('wrapped image', wrapped)
# # cv2.waitKey(0)
# img_size = (img.shape[1], img.shape[0])
# wrapped = cv2.warpPerspective(img, mat,img_size,cv2.INTER_LINEAR )
# cv2.imwrite('projectstages_out_images/prespective_test.png',img)
# cv2.imwrite('projectstages_out_images/wrapped.png',wrapped)


# testing region of intrest selection
# img = cv2.imread('test_images/test3.jpg',1)
# sub_img = utils.region_of_intrest(img)
# cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('region of intrest', cv2.WINDOW_NORMAL)
# cv2.imshow('original image', img)
# cv2.imshow('region of intrest', sub_img)
# cv2.waitKey(0)
# cv2.imwrite('projectstages_out_images/unmasked_test.png',img)
# cv2.imwrite('projectstages_out_images/masked.png',sub_img)


# image thersholding
# img = cv2.imread('test_images/test3.jpg')
# thresh = utils.image_threshold(img)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(thresh, cmap='gray')
# ax2.set_title('Combined', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
# plt.imsave('projectstages_out_images/thresholded.png',thresh)