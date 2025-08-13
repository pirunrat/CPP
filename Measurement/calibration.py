import cv2
import numpy as np
import argparse

# ===== Globals =====
measuring = False
click_points = []
H = None
frame_display = None
last_box_pts = None
last_width = None
last_height = None
# ===================

def calibrate_camera(objpoints, imgpoints, image_size):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print(f"[INFO] Calibration RMS error: {ret}")
    return K, dist, rvecs, tvecs

def compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
    total_err = 0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2).astype(np.float32)
        imgp = imgpoints[i].reshape(-1, 2).astype(np.float32)
        err = cv2.norm(imgp, proj, cv2.NORM_L2)
        total_err += err * err
        total_pts += len(objpoints[i])
    rmse = np.sqrt(total_err / total_pts) if total_pts > 0 else 0.0
    return rmse

def mouse_callback(event, x, y, flags, param):
    global click_points, measuring, frame_display
    if measuring and event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)  # dot immediately
        if len(click_points) == 4:
            measure_box(click_points)
            click_points = []

def measure_box(pts):
    global H, frame_display, last_box_pts, last_width, last_height
    img_pts = np.array([pts], dtype=np.float32)
    real_pts = cv2.perspectiveTransform(img_pts, H)[0]  # shape (4,2) in mm

    def dist_mm(p1, p2):
        return np.linalg.norm(p1 - p2)

    width = dist_mm(real_pts[0], real_pts[1])
    height = dist_mm(real_pts[1], real_pts[2])

    print(f"[MEASURE] Width: {width:.2f} mm, Height: {height:.2f} mm")

    last_box_pts = pts
    last_width = width
    last_height = height

def draw_last_box(frame):
    """Draw last measured box, dots, and dimensions."""
    if last_box_pts is not None:
        pts = last_box_pts
        cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{last_width:.1f} mm",
                    ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"{last_height:.1f} mm",
                    ((pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def draw_help_overlay(frame):
    """Draw command instructions on the left side."""
    help_text = [
        "[ c ] Capture chessboard",
        "[ k ] Calibrate",
        "[ p ] Perspective",
        "[ m ] Measure box",
        "[ u ] Toggle undistort",
        "[ r ] Reset",
        "[ q ] Quit"
    ]
    y0 = 25
    for i, line in enumerate(help_text):
        y = y0 + i * 25
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    global measuring, click_points, H, frame_display, last_box_pts

    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="Chessboard inner corners cols")
    ap.add_argument("--rows", type=int, default=6, help="Chessboard inner corners rows")
    ap.add_argument("--square", type=float, default=25.0, help="Square size in mm")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    args = ap.parse_args()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(args.cam)
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    K = None
    dist = None
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

        draw_last_box(frame_display)     # draw any measured box
        draw_help_overlay(frame_display) # draw help menu

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"[INFO] Captured view #{len(objpoints)}")

        elif key == ord('k') and len(objpoints) >= 8:
            h, w = gray.shape[:2]
            K, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, (w, h))
            err = compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)
            print(f"[INFO] Total reprojection error: {err:.4f} px")

        elif key == ord('u') and K is not None:
            undistort_mode = not undistort_mode

        elif key == ord('p') and found and K is not None:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_pts = corners2.reshape(-1, 2).astype(np.float32)
            H, _ = cv2.findHomography(img_pts, objp[:, :2])
            print("[INFO] Perspective homography computed.")

        elif key == ord('m'):
            if H is not None:
                measuring = True
                click_points = []
                print("Click 4 corners of the box (top-left → top-right → bottom-right → bottom-left)")
            else:
                print("[WARN] Run perspective mode (p) first.")

        elif key == ord('r'):
            objpoints.clear()
            imgpoints.clear()
            K = None
            dist = None
            last_box_pts = None
            print("[INFO] Reset calibration data.")

        elif key == ord('q'):
            break

        if undistort_mode and K is not None:
            frame_display = cv2.undistort(frame_display, K, dist)

        cv2.imshow("Calibration", frame_display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
