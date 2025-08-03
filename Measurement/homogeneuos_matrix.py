import cv2
import os

# Create a directory to save the captured images
output_dir = "calibration_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

img_count = 0
print("Press 'c' to capture an image and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Display the current frame
    cv2.imshow('Camera Feed', frame)
    
    key = cv2.waitKey(1)
    
    # Capture image when 'c' is pressed
    if key == ord('c'):
        img_name = os.path.join(output_dir, f"calibration_image_{img_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Image {img_name} saved.")
        img_count += 1
        
    # Quit when 'q' is pressed
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
print(f"Captured {img_count} images.")