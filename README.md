# Advanced_lane_detection

The project is done as part of Udacity self Driving Car Nano Degree Program.
Finding lane line in real video  in different lane and external conditions.

The following steps are performed 
1. calibrate the camera and undistorted the  video frame.
2. Masking to region of interest.
3. compute the perspective transform and warp the image.
4. use thresholding operation to find lane lines robustly.
5. use histogram and sliding window operations to find lane line pixel and fit a polynomial to it.
6. compute the lane curvature and distance of the vehicle from center of the lane.

The output looks something like this:


 
