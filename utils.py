## the script contains functions for image manipulation.
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# constants

corners_x = 9
corners_y = 6



def camera_calibration():

    object_pts = []
    image_pts = []

    calibartion_images = glob.glob('camera_cal/calibration*.jpg')

    for img in calibartion_images:
        cal_img = cv2.imread(img,1)
        cal_gray = cv2.cvtColor(cal_img,cv2.COLOR_BGR2GRAY)
        ret , corners = cv2.findChessboardCorners(cal_gray, (corners_x, corners_y), None)

        if ret:
            objpt = np.zeros((corners_x*corners_y,3), np.float32)
            objpt[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1,2)
            object_pts.append(objpt)
            image_pts.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_pts, image_pts, cal_gray.shape[::-1], None, None)
    return mtx, dist


def region_of_intrest(image):
# assuming camera is in the center of the car and car is trying to drive center of the lane

    bottom_left_x = image.shape[1] * .01
    bottom_left_y = image.shape[0]
    top_left_x = int(image.shape[1] * .35)
    top_left_y = int(image.shape[0] * .65)
    top_right_x = int(image.shape[1] * .65)
    top_right_y = int(image.shape[0] * .65)
    bottom_right_x = int(image.shape[1] + (image.shape[1] * .1))
    bottom_right_y = int(image.shape[0])


    vertices = np.array([[(bottom_left_x, bottom_left_y),
                          (top_left_x,top_left_y),
                          (top_right_x,top_right_y),
                          ( bottom_right_x, bottom_right_y)]], dtype=np.int32)

    # print(image.shape)
    # print (vertices)

    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image




def get_prespective(image, unwrapped = False):
    image_width = image.shape[1]
    image_height = image.shape[0]

    # Source coordinates
    src = np.float32([
        [image_width * 0.4, image_height * 0.65],
        [image_width * 0.5, image_height * 0.65],
        [image_width * 0.1, image_height * 0.95],
        [image_width * 0.8, image_height * 0.95],
    ])

    # Destination coordinates
    dst = np.float32([
        [image_width * 0.2, image_height * 0.025],
        [image_width * 0.8, image_height * 0.025],
        [image_width * 0.2, image_height * 0.97],
        [image_width * 0.8, image_height * 0.97],
    ])
    # print (src)
    # print ()
    # print (dst)
    if (unwrapped):
        src, dst = dst, src

    M = cv2.getPerspectiveTransform(src, dst)

    return M


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
  # grayscale image
  red = img[:, :, 0]

  # find abs sobel thresh
  if orient == 'x':
    sobel = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
    sobel = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

  #get abs value
  abs_sobel = np.absolute(sobel)
  scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

  grad_binary = np.zeros_like(scaled)
  grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
  return grad_binary


'''
calculate magnitude of gradient given an image and threshold
'''
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
  # gray scale
  red = img[:, :, 0]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  mag = np.sqrt(abs_x ** 2 + abs_y ** 2)
  scaled = (255*mag/np.max(mag))

  binary_output = np.zeros_like(scaled)
  binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
  return binary_output

'''
calculate direction of gradient given image and thresh
'''
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
  # red = img[:, :, 0]

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  sobel_dir = np.arctan2(abs_y, abs_x)

  binary_output = np.zeros_like(sobel_dir)
  binary_output[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
  return binary_output

'''
calculate the threshold of the hls values
'''
def hls_thresh(img, thresh=(0, 255)):
  hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

  s_channel = hls[:, :, 2]

  binary_output = np.zeros_like(s_channel)
  binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

  return binary_output

'''
get v channel from hsv
'''
def hsv_thresh(img, thresh=(0, 255)):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  v_channel = hsv[:, :, 2]

  binary_output = np.zeros_like(v_channel)
  binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1

  return binary_output


def find_white_lane(img, thresh= (0,255)):
    xyz = cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
    z_channel = xyz[:,:,2]
    binary_output = np.zeros_like(z_channel)
    binary_output[(z_channel > thresh[0]) & (z_channel <= thresh[1])] = 1
    return binary_output


def image_threshold(img):
    x_thresholded = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 120))

    y_thresholded = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 100))


    hls_thresholded = hls_thresh(img, thresh=(100, 255))

    hsv_thresholded = hsv_thresh(img, thresh=(50, 255))


    dir_thresholded = dir_thresh(img, sobel_kernel=15, thresh=(.7, 1.2))

    white_lane = find_white_lane(img, thresh =(103, 193) )

    mag_thresholded = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))



    binary_output = np.zeros_like(dir_thresholded)
    binary_output[((hsv_thresholded == 1) & (hls_thresholded == 1)) | ((x_thresholded == 1) & (y_thresholded == 1))  ] = 1


    return binary_output


test_img = cv2.imread('test_images/test5.jpg')
output = image_threshold(test_img)