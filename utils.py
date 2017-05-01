## the script contains functions for image manipulation.
import cv2
import numpy as np
import glob

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


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient

    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 5) Return this mask as your binary_output image
    return sxbinary

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def image_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    light_mask = np.zeros_like(l_channel)
    light_mask[(s_channel >= 5) & (l_channel >= 150)] = 1
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx_l = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(25, 100))
    gradx_s = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(10, 100))


    combined = np.zeros_like(gradx_s)
    combined[((gradx_l == 1) | (gradx_s == 1)) & (light_mask == 1)] = 1
    return combined


