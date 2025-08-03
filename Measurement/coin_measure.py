import numpy as np
import cv2

# Load the saved camera parameters and scale factor
try:
    with np.load('camera_params.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    with np.load('scale_factor.npz') as data:
        pixels_per_cm = data['pixels_per_cm']
    print("Calibration and scale parameters loaded successfully.")
    print(f"Pixels per CM: {pixels_per_cm:.2f}")
except FileNotFoundError:
    print("Error: Required files not found. Please run the calibration scripts first.")
    exit()

# Set up video capture and trackbars
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create a window for the trackbars and a callback function
cv2.namedWindow("Settings")
def nothing(x):
    pass

# Create trackbars for Hough Circle parameters
# Adjust these values to fine-tune circle detection for your lighting and coin size
cv2.createTrackbar("dp", "Settings", 1, 2, nothing) # Inverse ratio of accumulator resolution
cv2.createTrackbar("minDist", "Settings", 50, 200, nothing) # Minimum distance between circles
cv2.createTrackbar("param1", "Settings", 200, 300, nothing) # Upper threshold for Canny edge detector
cv2.createTrackbar("param2", "Settings", 100, 200, nothing) # Accumulator threshold for circle centers
cv2.createTrackbar("minRadius", "Settings", 20, 100, nothing) # Minimum radius in pixels
cv2.createTrackbar("maxRadius", "Settings", 100, 200, nothing) # Maximum radius in pixels
cv2.createTrackbar("Median Blur", "Settings", 5, 25, nothing)

print("Press 'q' to quit.")
print("Adjust the sliders in the 'Settings' window to tune circle detection.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, mtx, dist, None)
    
    # Pre-process the image for better object detection
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    
    # Get median blur value and ensure it's odd
    median_blur_val = cv2.getTrackbarPos("Median Blur", "Settings")
    median_blur_val = max(3, median_blur_val | 1)
    blurred = cv2.medianBlur(gray, median_blur_val)

    # Get trackbar values on the fly
    dp = cv2.getTrackbarPos("dp", "Settings")
    minDist = cv2.getTrackbarPos("minDist", "Settings")
    param1 = cv2.getTrackbarPos("param1", "Settings")
    param2 = cv2.getTrackbarPos("param2", "Settings")
    minRadius = cv2.getTrackbarPos("minRadius", "Settings")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Settings")

    # Use a minimum value of 1 to avoid errors
    if dp == 0: dp = 1

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, 
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Iterate through the detected circles
        for i in circles[0, :]:
            # Get the center (x, y) and radius
            center_x, center_y, radius = i[0], i[1], i[2]
            
            # Calculate the diameter in centimeters
            diameter_pixels = 2 * radius
            diameter_cm = diameter_pixels / pixels_per_cm
            
            # Draw the circle and measurement
            cv2.circle(undistorted_frame, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(undistorted_frame, (center_x, center_y), 2, (0, 0, 255), 3) # Draw center
            
            text = f"Diameter: {diameter_cm:.2f} CM"
            cv2.putText(undistorted_frame, text, (center_x - 50, center_y - radius - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the final frame
    cv2.imshow('Live Measurement', undistorted_frame)
    cv2.imshow('Blurred Image', blurred)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()