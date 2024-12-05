**Lane Finding Project**


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Image References:

1. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Original%20Chessboard%20Image.PNG?raw=true
2. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Undistorted%20Chessboard%20Image.PNG?raw=true
3. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Original%20Image.PNG?raw=true
4. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Undistored%20Image.PNG?raw=true
5. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Binary%20Image.PNG?raw=true
6. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Wraped%20Image%20(Perspective%20Change).PNG?raw=true
7. https://github.com/Lara2901998/ProjectOpenCV/blob/main/Line%20Detection%20on%20Road.PNG?raw=true


Before running the program, you need to download the main.py file and place it in the project's root folder. The image paths are defined in the code according to the initial folder provided.
The program will display images in the following order: first the chessboard image, then the undistorted version, followed by the test image and its undistorted version, then edge detection, perspective transformation, and finally lane detection with lines. The example is for one image; to check another image, you need to uncomment the path to that image.


Camera Calibration

To calculate the camera matrix and distortion coefficients, I used a set of chessboard images. I applied the function cv2.findChessboardCorners() to detect the chessboard corners, which were then used to estimate the camera calibration matrix (mtx) and distortion coefficients (dist) using the function cv2.calibrateCamera(). The calibration process is essential for correcting image distortion caused by the camera lens. Below is an example of an image with corrected distortion:
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Original%20Chessboard%20Image.PNG?raw=true
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Undistorted%20Chessboard%20Image.PNG?raw=true
Here is the original image with corrected distortion after applying camera calibration:
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Original%20Image.PNG?raw=true
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Undistored%20Image.PNG?raw=true


Creating a binary image

In the code, I used color transformations to create a binary image with a threshold. The input image is first converted to HSV color (cv2.cvtColor()), which makes color selection easier. Then, two masks were created to extract yellow and white lanes using the cv2.inRange() function. These masks were combined using the bitwise OR operator, and edge detection was applied using the Canny algorithm after Gaussian blurring to reduce noise.

hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

yellow_mask = cv2.inRange(hsv, (15, 120, 200), (50, 255, 255))

white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 25, 255))

mask = cv2.bitwise_or(yellow_mask, white_mask)

edges = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 0), 50, 150)

Here is an example of the result of the binary image:

https://github.com/Lara2901998/ProjectOpenCV/blob/main/Binary%20Image.PNG?raw=true


Perspective transformation

I applied a perspective transformation to convert the image into a "bird's eye view," which helps in better lane line detection. I defined the source and destination points (pts1 and pts2) and applied the perspective transformation using cv2.getPerspectiveTransform() and cv2.warpPerspective().

matrix = cv2.getPerspectiveTransform(pts1, pts2)

warped_image = cv2.warpPerspective(roi_edges, matrix, (width, height))

Here is an example of the transformed image after applying the perspective transformation:
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Wraped%20Image%20(Perspective%20Change).PNG?raw=true


Detection of lane pixels

For lane pixel detection, I used a method that identifies all non-zero pixels in the binary transformed image (warped_image.nonzero()). Then, I applied a condition to separate the pixels of the left and right lanes and fitted a polynomial using np.polyfit() for those points. The result is a polynomial that best approximates the left and right lane lines.

left_condition = (x_coords > width // 4) & (x_coords < width // 2)

right_condition = (x_coords > width // 2) & (x_coords < 3 * width // 4)

fit_left = np.polyfit(y_coords[left_condition], x_coords[left_condition], 1)

fit_right = np.polyfit(y_coords[right_condition], x_coords[right_condition], 1)

Determining curvature and vehicle position

The radius of curvature for both lanes is calculated using the polynomial coefficients obtained from the lane fitting. A formula is used that takes into account the pixel positions and provides the curvature in meters. Additionally, the vehicle's position relative to the lane center is calculated by determining the distance from the lane center to the vehicle's position.

left_curvature = ((1 + (2 * fit_left[0] * height + fit_left[1])**2)**(3/2)) / np.abs(2 * fit_left[0])

right_curvature = ((1 + (2 * fit_right[0] * height + fit_right[1])**2)**(3/2)) / np.abs(2 * fit_right[0])

lane_center = (left_line[height - 1] + right_line[height - 1]) / 2

vehicle_offset = (lane_center - width / 2) * 3.7 / width


Displaying results

Here is an example of the final image where the lane boundaries are visible, along with the curvature and the vehicle's position:
https://github.com/Lara2901998/ProjectOpenCV/blob/main/Line%20Detection%20on%20Road.PNG?raw=true


Video Processing

Currently, the pipeline is not implemented for video recordings. Before video processing can be implemented, lane detection needs to be improved, as the current method is not precise enough for dynamic environments. The problem with curvature arises because the lane detection is not advanced enough, so in the future, the RANSAC method or Hough transform should be implemented. Currently, the implementation is functional for static images.


Discussion

The current implementation works for processing static images, but video processing has not been implemented yet. To enable video processing, the code needs to be improved for better precision and reliability, especially in lane detection and tracking.

During the project, I faced several challenges. The first issue was with the perspective transformation. It was difficult to adjust the parameters to get the correct "bird's-eye view" of the image. Even though the transformation was successful, some imperfections and inaccuracies remained, which affected the lane detection accuracy.

The second challenge was lane detection. I did not use advanced techniques like the RANSAC method or Hough transform (I had problems implementing these methods, so I used a simpler linear fitting method for this, although it's definitely less accurate), which are more effective for detecting lines, especially in dynamic conditions like video. Without these methods, lane tracking and curvature calculations may not be as accurate, which is a problem for video processing.

The third challenge was combining all the results into the final image. Even though I followed all the necessary steps, aligning the lane detection, perspective transformation, and final image was tricky. It became especially challenging when trying to merge the detected lane boundaries with the original road image.

To make the pipeline more robust, I need to use advanced methods like RANSAC and Hough transform for better lane tracking and curvature calculations. Also, dynamically adjusting color thresholds would help improve accuracy in different lighting conditions and driving environments.

Right now, the images test1, test3, and solidYellowCurve do not detect lanes properly due to excessive brightness, road irregularities, or uneven lane colors. These factors make it hard to distinguish the lanes in the images. On the other hand, the pipeline works well for the other test images, where lighting and lane colors are more consistent. Improvements are needed to adjust the pipeline for varying lighting and color conditions to ensure accuracy in all situations.

