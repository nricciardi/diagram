Scrivo qui - o Nico - per semplicità:

```python
import cv2
import numpy as np

sift = cv2.SIFT_create()

# Get template AND image (bbox of the arrow)
kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(image, None)

# Find a match of the keypoints' descriptors (NOTE: Chat suggest to also use FLANN)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < RATIO_THRESHOLD * n.distance:
        good.append(m)

# Compute the homography - finds the matches
if len(good) > 1:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
    matchesMask = mask.ravel().tolist()
else:
    # Repeat with a smaller RATIO_THRESHOLD?

h, w = template.shape[:2]

template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# Get the corners of the template in the image
projected_corners = cv2.perspectiveTransform(template_corners, M)
```

NOTA: Questo template matching è scale-invariant e rotation-invariant. Così si hanno i quattro punti del template trovato sull'immagine.