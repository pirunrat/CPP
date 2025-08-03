import numpy as np
import cv2

def haar_1d_forward(signal):
    n = len(signal)
    output = signal.copy().astype(float)
    result = []
    while n > 1:
        temp = []
        for i in range(0, n, 2):
            avg = (output[i] + output[i+1]) / np.sqrt(2)
            diff = (output[i] - output[i+1]) / np.sqrt(2)
            temp.append(avg)
            result.append(diff)
        output[:n//2] = temp
        n //= 2
    return np.concatenate((output[:1], result))

def haar_1d_inverse(coeffs):
    coeffs = coeffs.copy().astype(float)
    n = 1
    while n * 2 <= len(coeffs):
        temp = []
        for i in range(n):
            avg = coeffs[i]
            diff = coeffs[n + i]
            a = (avg + diff) / np.sqrt(2)
            b = (avg - diff) / np.sqrt(2)
            temp.extend([a, b])
        coeffs[:n*2] = temp
        n *= 2
    return coeffs

def haar_2d_forward(image):
    h, w = image.shape
    output = image.copy().astype(float)
    
    for row in range(h):
        output[row] = haar_1d_forward(output[row])
    
    for col in range(w):
        output[:, col] = haar_1d_forward(output[:, col])
    
    return output

def haar_2d_inverse(coeffs):
    h, w = coeffs.shape
    output = coeffs.copy().astype(float)
    
    for col in range(w):
        output[:, col] = haar_1d_inverse(output[:, col])
    
    for row in range(h):
        output[row] = haar_1d_inverse(output[row])
    
    return output

# ✅ Read image and convert to grayscale
img = cv2.imread('./Test/download.jpg', cv2.IMREAD_GRAYSCALE)

# ✅ Resize to power of 2 (e.g., 256x256)
img = cv2.resize(img, (256, 256))

# ✅ Apply Haar transform
coeffs_2d = haar_2d_forward(img)
reconstructed_img = haar_2d_inverse(coeffs_2d)

# ✅ Normalize coefficients for display (because they can be negative)
coeffs_display = cv2.normalize(coeffs_2d, None, 0, 255, cv2.NORM_MINMAX)
coeffs_display = coeffs_display.astype(np.uint8)

reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)

# ✅ Show images using OpenCV
cv2.imshow("Original Image", img)
cv2.imshow("Haar Coefficients", coeffs_display)
cv2.imshow("Reconstructed Image", reconstructed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
