import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('/home/lester/Desktop/test_img.png', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

img_array = []
size = None
while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    good_matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    # good_matches = matches[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    dst += (w, 0)  # adding offset

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, frame, kp2, good_matches, None, **draw_params)
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    # print(dst.shape)
    # img3 = cv2.drawMatches(img1, kp1, frame, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("test", img3)
    img_array.append(img3)
    if not size:
        h, w, c = img3.shape
        size = (w, h)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter('/home/lester/Desktop/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()