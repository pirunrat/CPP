import cv2
import numpy as np
import argparse

# ===== Globals =====
click_points = []
last_box_pts = None
last_width = None
last_height = None
frame_display = None
scale_mm_per_pixel = None
K = None
dist = None
# ===================

def calibrate_camera(objpoints, imgpoints, image_size):
    global K, dist
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print(f"[INFO] Calibration RMS error: {ret}")
    return K, dist, rvecs, tvecs

def mouse_callback(event, x, y, flags, param):
    global click_points, frame_display, last_box_pts, last_width, last_height
    if event == cv2.EVENT_LBUTTONDOWN and scale_mm_per_pixel is not None:
        click_points.append((x, y))
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
        if len(click_points) == 4:
            pts = np.array(click_points, dtype=np.float32)
            width = np.linalg.norm(pts[0]-pts[1]) * scale_mm_per_pixel
            height = np.linalg.norm(pts[1]-pts[2]) * scale_mm_per_pixel
            print(f"[MEASURE] Width: {width:.1f} mm, Height: {height:.1f} mm")
            last_box_pts = click_points.copy()
            last_width = width
            last_height = height
            click_points.clear()

def draw_last_box(frame):
    if last_box_pts is not None:
        pts = last_box_pts
        cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 5, (0,0,255), -1)
        cv2.putText(frame, f"{last_width:.1f} mm",
                    ((pts[0][0]+pts[1][0])//2, (pts[0][1]+pts[1][1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"{last_height:.1f} mm",
                    ((pts[1][0]+pts[2][0])//2, (pts[1][1]+pts[2][1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def draw_help_overlay(frame):
    help_text = [
        "[ c ] Capture chessboard",
        "[ k ] Calibrate",
        "[ m ] Measure object (click 4 points)",
        "[ u ] Toggle undistort",
        "[ r ] Reset",
        "[ q ] Quit"
    ]
    y0 = 25
    for i, line in enumerate(help_text):
        y = y0 + i*25
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def compute_scale(imgpoints, square_size):
    """Compute mm per pixel using undistorted first chessboard image."""
    pts = imgpoints[0].reshape(-1,2)
    # Compute distances along X and Y (horizontal and vertical squares)
    cols = int(np.sqrt(len(pts)))  # assume square grid
    dxs = [np.linalg.norm(pts[i]-pts[i+1]) for i in range(len(pts)-1) if (i+1)%cols !=0]
    dys = [np.linalg.norm(pts[i]-pts[i+cols]) for i in range(len(pts)-cols)]
    mean_px = (np.mean(dxs)+np.mean(dys))/2
    scale = square_size / mean_px
    return scale

def main():
    global frame_display, scale_mm_per_pixel, K, dist, last_box_pts

    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((args.rows*args.cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:args.cols,0:args.rows].T.reshape(-1,2)*args.square

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(args.cam)
    cv2.namedWindow("Top-down Measurement")
    cv2.setMouseCallback("Top-down Measurement", mouse_callback)

    undistort_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        frame_display = frame.copy()

        if found:
            cv2.drawChessboardCorners(frame_display, (args.cols, args.rows), corners, found)

        draw_last_box(frame_display)
        draw_help_overlay(frame_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"[INFO] Captured view #{len(objpoints)}")

        elif key == ord('k') and len(objpoints) >= 8:
            h, w = gray.shape[:2]
            K, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, (w,h))
            # Undistort first image for scale computation
            undistorted = cv2.undistort(imgpoints[0], K, dist)
            scale_mm_per_pixel = compute_scale([undistorted], args.square)
            undistort_mode = True
            print(f"[INFO] Scale: {scale_mm_per_pixel:.3f} mm/pixel")

        elif key == ord('u') and K is not None:
            undistort_mode = not undistort_mode

        elif key == ord('r'):
            objpoints.clear()
            imgpoints.clear()
            K = None
            dist = None
            last_box_pts = None
            scale_mm_per_pixel = None
            print("[INFO] Reset data")

        elif key == ord('q'):
            break

        if undistort_mode and K is not None:
            frame_display = cv2.undistort(frame_display, K, dist)

        cv2.imshow("Top-down Measurement", frame_display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
