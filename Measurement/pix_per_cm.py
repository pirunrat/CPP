import numpy as np
import cv2

# Define the dimensions of the chessboard
chessboard_size = (9, 6)
square_size_mm = 25.0  # Real-world size of a chessboard square in mm
square_size_cm = square_size_mm / 10.0 # Convert to cm

# Object points for solvePnP (if needed later)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size_cm

# Load the saved camera parameters
try:
    with np.load('camera_params.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: 'camera_params.npz' not found. Please run the initial camera calibration script first.")
    exit()

# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Place the chessboard at your desired working distance.")
print("Press 's' to save the scale factor, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = cv2.undistort(frame, mtx, dist, None)
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    scale_factor_text = "Scale not yet calculated..."
    
    if ret:
        # Get the pixel distance between two adjacent squares
        pixel_dist_x = np.sqrt(np.sum((corners[1][0] - corners[0][0])**2))
        pixel_dist_y = np.sqrt(np.sum((corners[chessboard_size[0]][0] - corners[0][0])**2))
        
        avg_pixel_dist = (pixel_dist_x + pixel_dist_y) / 2
        pixels_per_cm = avg_pixel_dist / square_size_cm
        
        scale_factor_text = f"Scale: {pixels_per_cm:.2f} px/cm"

        # Draw the corners to visualize the detection
        cv2.drawChessboardCorners(undistorted_frame, chessboard_size, corners, ret)

    # Display the live feed with the scale information
    cv2.putText(undistorted_frame, scale_factor_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Scale Calibration', undistorted_frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s') and ret:
        # Save the scale factor and exit
        np.savez('scale_factor.npz', pixels_per_cm=pixels_per_cm)
        print(f"\nScale factor saved: {pixels_per_cm:.2f} px/cm")
        print("You can now run the measurement script.")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()