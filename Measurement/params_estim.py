import numpy as np
import cv2

# Define the dimensions of the chessboard
chessboard_size = (9, 6)
square_size = 25  # in millimeters

# Object points, will be the same for all images
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []
imgpoints = []
image_frames = []

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initialize a counter for the number of captured images
calibration_images_count = 0

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
    
# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("Press 's' to save a calibration image.")
print("Press 'q' to quit and start calibration.")

# --- Capture frames and detect corners ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing to improve corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance local contrast
    enhanced_gray = clahe.apply(gray)
    
    # Use adaptive thresholding and other flags for more robust detection
    ret, corners = cv2.findChessboardCorners(enhanced_gray, chessboard_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # Provide visual feedback
    font = cv2.FONT_HERSHEY_SIMPLEX
    if ret:
        corners2 = cv2.cornerSubPix(enhanced_gray, corners, (11, 11), (-1, -1), criteria)
        cv2.putText(frame, "Corners Found! Press 's' to save.", (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
    else:
        cv2.putText(frame, "Corners NOT Found. Adjust view.", (20, 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Calibration View', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners2)
            image_frames.append(frame.copy())
            calibration_images_count += 1
            print(f"Image {calibration_images_count} captured.")
        else:
            print("No chessboard corners detected. Please try again.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Calibrate and filter bad images ---
if calibration_images_count > 10:
    print("Starting initial calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    reprojection_errors = []
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        reprojection_errors.append(error)
    
    mean_error = np.mean(reprojection_errors)
    std_dev = np.std(reprojection_errors)
    dynamic_threshold = mean_error + 1.5 * std_dev
    
    print(f"Mean Reprojection Error: {mean_error:.4f}, Std Dev: {std_dev:.4f}")
    print(f"Dynamic Threshold set to: {dynamic_threshold:.4f}")

    valid_objpoints = []
    valid_imgpoints = []
    
    print("Filtering images based on reprojection error...")
    
    for i in range(len(objpoints)):
        error = reprojection_errors[i]
        if error < dynamic_threshold:
            valid_objpoints.append(objpoints[i])
            valid_imgpoints.append(imgpoints[i])
            print(f"Image {i+1}: Reprojection Error: {error:.4f} (Kept)")
        else:
            print(f"Image {i+1}: Reprojection Error: {error:.4f} (Discarded)")

    if len(valid_objpoints) > 5:
        print(f"\nRecalibrating with {len(valid_objpoints)} valid images...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(valid_objpoints, valid_imgpoints, gray.shape[::-1], None, None)
        final_mean_error = 0
        for i in range(len(valid_objpoints)):
            imgpoints_reprojected, _ = cv2.projectPoints(valid_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(valid_imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
            final_mean_error += error
        final_mean_error /= len(valid_objpoints)
        
        # print("mtx shape:", mtx.shape)       # Camera matrix
        # print("dist shape:", dist.shape)     # Distortion coefficients
        # print("Number of rvecs:", len(rvecs))
        # print("rvec[0] shape:", rvecs[0].shape if len(rvecs) > 0 else None)
        # print("Number of tvecs:", len(tvecs))
        # print("tvec[0] shape:", tvecs[0].shape if len(tvecs) > 0 else None)

        print("\nFinal Calibration successful.")
        print(f"Final Reprojection Error: {final_mean_error:.4f}")
        print("\nCamera Matrix:\n", mtx)
        print(f'\nRvecs : {rvecs}\n')
        print(f'\nTvecs : {tvecs}\n')
        print("\nDistortion Coefficients:\n", dist)
    
        np.savez('camera_params.npz', mtx=mtx, dist=dist)
        print("\nCalibration parameters saved to camera_params.npz")
    else:
        print("\nToo few valid images remain after filtering. Calibration aborted.")

else:
    print("Not enough images captured for calibration. Please capture at least 10 images.")