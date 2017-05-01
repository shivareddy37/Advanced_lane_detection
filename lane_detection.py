import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from moviepy.editor import VideoFileClip

# constants for lanes ,US standard
YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension


def lane_curvature(leftx, lefty, rightx, righty):
    left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    left_curve_rad = ((1 + (2 * left_fit_cr[0] * np.max(lefty) + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curve_rad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    return left_curve_rad, right_curve_rad


def dist_from_center(frame, left_fit, right_fit):
    vehicle_position = frame.shape[1] / 2
    center_of_lane = (left_fit[-1] + right_fit[-1]) // 2
    center_calc = (vehicle_position - center_of_lane) * XM_PER_PIX
    return center_calc


def process(frame):
    # steps
    # 1. select region of intrest
    # 2. undistort the image
    # 3. compute the prepective transform and warp image
    # 4. find the thresholded image
    # 5. create histogram and then find line pixels by anaylsing peaks and sliding window technique


    roi = utils.region_of_intrest(frame)
    undistorted_img = cv2.undistort(roi,cam_mat, dist_coff, None, cam_mat)
    img_size = (undistorted_img.shape[1], undistorted_img.shape[0])
    trans_mat = utils.get_prespective(undistorted_img)
    inv_trans_mat = utils.get_prespective(undistorted_img, unwrapped = True)
    warped_img = cv2.warpPerspective(undistorted_img, trans_mat,img_size,cv2.INTER_LINEAR )
    thresh_img = utils.image_threshold(warped_img)



    # histogram to find peaks  for staring points of lane detection

    histogram = np.sum(thresh_img[int(thresh_img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((thresh_img, thresh_img, thresh_img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    window_height = np.int(thresh_img.shape[0] / nwindows)
    nonzero = thresh_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = thresh_img.shape[0] - (window + 1) * window_height
        win_y_high = thresh_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, thresh_img.shape[0] - 1, thresh_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    # calulation the lane curvature and distance from center

    left_curverad, right_curverad = lane_curvature(leftx, lefty, rightx, righty)
    center_calc = dist_from_center(frame, left_fitx, right_fitx)

    warp_zero = np.zeros_like(thresh_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # this is a hamfisted solution to replace curves with extreme values with the previous frame's curve

    newwarp = cv2.warpPerspective(color_warp, inv_trans_mat, (frame.shape[1], frame.shape[0]))

    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

    text = 'curvature radius: {0} m. '.format((int(left_curverad) + int(right_curverad)) / 2)
    text2 = 'distance from center: {0} m. '.format((np.math.ceil(abs(center_calc) * 100) / 100))
    cv2.putText(result, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, text2, (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA)



    return result




cam_mat, dist_coff =  utils.camera_calibration()

# test_img = cv2.imread('test_images/test3.jpg')lane_curvature and dist130
# output = process(test_img)
# cv2.imwrite('final_output.png', output)

video = VideoFileClip('project_video.mp4')
input = video.fl_image(process)
input.write_videofile('project_video_output.mp4', audio = False)

