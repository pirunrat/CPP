import numpy as np
import cv2

# Define the dimensions of the chessboard and the size of the squares
chessboard_size = (9, 6)
square_size = 25  # in millimeters

# Object points for solvePnP
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# --- Load the saved camera parameters ---
try:
    with np.load('camera_params.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: 'camera_params.npz' not found. Please run the calibration script first.")
    exit()

# --- Capture a live feed and measure the distance ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'q' to quit the measurement.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = cv2.undistort(frame, mtx, dist, None)
    gray_undistorted = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray_undistorted, chessboard_size, None)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0) # Green color

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray_undistorted, corners, (11, 11), (-1, -1), criteria)
        
        # Use solvePnP to find the object's 3D pose
        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        # Get the Z-distance for scaling
        distance_to_object_mm = tvec[2][0]

        # Get the focal length from the camera matrix
        focal_length_pixels = mtx[0, 0]

        # Calculate a scale factor based on the known real-world size and measured pixel size
        # This is a simplification, but effective for this visualization
        scale_factor = square_size / (np.sqrt(np.sum((corners2[1][0] - corners2[0][0])**2)))
        
        # Now, iterate through each square to measure and display
        for i in range(chessboard_size[1] - 1):
            for j in range(chessboard_size[0] - 1):
                # Get the four corners for the current square
                p1 = corners2[i * chessboard_size[0] + j][0]
                p2 = corners2[i * chessboard_size[0] + j + 1][0]
                p3 = corners2[(i + 1) * chessboard_size[0] + j][0]
                p4 = corners2[(i + 1) * chessboard_size[0] + j + 1][0]
                
                # Calculate the side lengths in pixels
                side_horizontal = np.sqrt(np.sum((p2 - p1)**2))
                side_vertical = np.sqrt(np.sum((p3 - p1)**2))
                avg_pixel_dist = (side_horizontal + side_vertical) / 2
                
                # Convert pixel distance to millimeters using the scaling factor
                measured_mm = avg_pixel_dist * scale_factor

                # Draw a rectangle around the square
                pts = np.array([p1, p2, p4, p3], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(undistorted_frame, [pts], True, (0, 255, 0), 2)
                
                # Display the measured size in mm at the center of the square
                center_x = int((p1[0] + p4[0]) / 2)
                center_y = int((p1[1] + p4[1]) / 2)
                cv2.putText(undistorted_frame, f"{measured_mm:.1f} mm", (center_x - 30, center_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the overall distance to the chessboard
        #cv2.putText(undistorted_frame, f"Overall Distance: {distance_to_object_mm:.2f} mm", (50, 50), font, 1, text_color, 2, cv2.LINE_AA)
        
    else:
        cv2.putText(undistorted_frame, "Chessboard not found", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Live Measurement', undistorted_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()