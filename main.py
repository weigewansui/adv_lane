import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from tempfile import TemporaryFile

def write_output_img(img, img_name):
    name = 'CarND-Advanced-Lane-Lines/output_images/'+img_name
    cv2.imwrite(name, img)
# prepare object points
# nx = 9#TODO: enter the number of inside corners in x
# ny = 5#TODO: enter the number of inside corners in y
# images = glob.glob('CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')
# cali_name_list = range(1,20)
# img_list = []
# objpoints = []
# imgpoints = []
# objp = np.zeros((nx*ny, 3), np.float32)
# objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
#
# for fname in images:
#     # Make a list of calibration images
#     img = cv2.imread(fname)
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#     # If found, draw corners
#     if ret == True:
#         # Draw and display the corners
#         cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#         write_output_img(img, 'draw_corners.jpg')
#         imgpoints.append(corners)
#         objpoints.append(objp)
#
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# np.savez('tmp_distortion', mtx, dist)

npzfile = np.load('tmp_distortion.npz')
mtx = npzfile['arr_0']
dist = npzfile['arr_1']

test_img = cv2.imread('CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')
dst = cv2.undistort(test_img, mtx, dist, None, mtx)
write_output_img(dst, 'undistort_test_img.jpg')

hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS)
s_channel = hls[:,:,2]

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
scaled_sobel = np.absolute(sobel_x)

thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# color threshold
s_thresh_min = 140
s_thresh_max = 255

s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.set_title('Stacked thresholds')
# ax1.imshow(color_binary)
#
# ax2.set_title('Combined S channel and gradient thresholds')
# ax2.imshow(combined_binary, cmap='gray')
# plt.show()
# unwarp
src_pts = np.float32(
[
    [293, 661],
    [1003, 653],
    [515, 506],
    [793, 519]
]
)

dest_pts = np.float32(
[
    [293, 661],
    [1003, 653],
    [293, 506],
    [1003, 519]
]
)

M = cv2.getPerspectiveTransform(src_pts, dest_pts)
Minv = cv2.getPerspectiveTransform(dest_pts, src_pts)

# test on original image
img_size = (test_img.shape[1], test_img.shape[0])
warped = cv2.warpPerspective(test_img, M, img_size, flags=cv2.INTER_LINEAR)
write_output_img(warped, "warped_test_original.jpg")

tmp_image = combined_binary
img_size = (tmp_image.shape[1], tmp_image.shape[0])
binary_warped = cv2.warpPerspective(tmp_image, M, img_size, flags=cv2.INTER_LINEAR)
# plt.imshow(warped, cmap='gray')
# plt.show()

histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# plt.plot(histogram)
# plt.show()

out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# print(out_img.shape)
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []


# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 1)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 1)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# plt.imshow(out_img)
# plt.show()
# write_output_img(out_img, "bounding_box.jpg")
# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
write_output_img(out_img, "bounding_box_with_color.jpg")

out_img[ploty.astype(int), left_fitx.astype(int)] = [0,255,255]
out_img[ploty.astype(int), right_fitx.astype(int)] = [0,255,255]

write_output_img(out_img, "bounding_box_with_line.jpg")

# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
# plt.imshow(result)
# plt.show()
write_output_img(result, "line_with_window.jpg")
