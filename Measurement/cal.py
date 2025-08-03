import cv2
import numpy as np
import glob
import os

def detect_corners(images_path, checkerboard_size, square_size):
    """
    Detects checkerboard corners in a set of images.
    
    Args:
        images_path (str): Path to the directory containing calibration images.
        checkerboard_size (tuple): (width, height) of inner corners.
        square_size (float): The physical size of a single square (e.g., in mm).
        
    Returns:
        tuple: A tuple containing lists of object points and image points.
    """
    CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT = checkerboard_size
    
    # Prepare object points (3D coordinates of the corners in the world frame)
    objp = np.zeros((CHECKERBOARD_WIDTH * CHECKERBOARD_HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_WIDTH, 0:CHECKERBOARD_HEIGHT].T.reshape(-1, 2) * square_size
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    images = glob.glob(os.path.join(images_path, '*.jpg'))
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            
            # Optional: draw and display corners to verify
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(50)
            
    cv2.destroyAllWindows()
    
    return objpoints, imgpoints, gray.shape[::-1]

def calibrate_camera(objpoints, imgpoints, image_size):
    """
    Calibrates the camera using detected corner points.
    
    Args:
        objpoints (list): List of 3D object points.
        imgpoints (list): List of 2D image points.
        image_size (tuple): Dimensions of the image (width, height).
        
    Returns:
        tuple: A tuple containing the camera matrix, distortion coefficients,
               rotation vectors, and translation vectors.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    
    print("Camera Matrix (Intrinsic Parameters):\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    
    return mtx, dist, rvecs, tvecs

def get_homogeneous_matrix(rvec, tvec):
    """
    Constructs the 4x4 homogeneous matrix from a rotation vector and translation vector.
    
    Args:
        rvec (np.array): 3x1 rotation vector.
        tvec (np.array): 3x1 translation vector.
        
    Returns:
        np.array: The 4x4 homogeneous matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = tvec.flatten()
    
    return homogeneous_matrix

if __name__ == '__main__':
    # --- Configuration ---
    CHECKERBOARD_WIDTH = 9
    CHECKERBOARD_HEIGHT = 6
    CHECKERBOARD_SQUARE_SIZE = 25  # in mm
    IMAGES_PATH = 'calibration_images'
    
    # --- Step 1: Detect corners from images ---
    print("Step 1: Detecting checkerboard corners...")
    objpoints, imgpoints, image_size = detect_corners(IMAGES_PATH, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), CHECKERBOARD_SQUARE_SIZE)
    print("Found corners in {} images.".format(len(objpoints)))
    
    # --- Step 2: Calibrate the camera ---
    if len(objpoints) > 0:
        print("\nStep 2: Calibrating the camera...")
        mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)
        
        # --- Step 3: Get the Homogeneous Matrix for a specific pose ---
        print("\nStep 3: Calculating the homogeneous matrix for the first pose...")
        # We can get the homogeneous matrix for any of the captured images
        rvec_first = rvecs[0]
        tvec_first = tvecs[0]
        
        homogeneous_matrix_first_pose = get_homogeneous_matrix(rvec_first, tvec_first)
        print("Homogeneous Matrix:\n", homogeneous_matrix_first_pose)
        
        # --- Example of usage ---
        world_point = np.array([0, 0, 0, 1]).reshape(4, 1)
        camera_point = homogeneous_matrix_first_pose @ world_point
        print("\nWorld point (0,0,0) in camera coordinates:\n", camera_point[:3].flatten())