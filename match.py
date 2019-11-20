import cv2
import pafy
import numpy as np

proj_size = 1024
cv2.namedWindow("show1")
cv2.namedWindow("show2")
cam = cv2.VideoCapture(0)
cam_w, cam_h = 320, 240

url = 'https://youtu.be/0usXEzuKdkU'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
cap = cv2.VideoCapture(play.url)

while True:
    _, frame = cam.read()
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img1 = frame
        print("Photo Taken!")
cv2.destroyAllWindows()

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
proj_centroid = np.array([0, 0])

while True:
    try:
        _, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        good_matches = bf.match(des1, des2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        centroid = (dst_pts.sum(0) / len(dst_pts)).squeeze().astype('int')
        frame = cv2.circle(frame, tuple(centroid), 10, (255, 255, 255), -1)
        offset = np.array([cam_w//2, cam_h//2]) - centroid
        curr_centroid = np.array([proj_size//2, proj_size//2]) - offset
        if np.sum(np.abs(curr_centroid - proj_centroid)) >= 50:
            proj_centroid = curr_centroid
            x1, y1 = proj_centroid - proj_size//4
            x2, y2 = proj_centroid + proj_size//4
        projector = np.zeros((proj_size, proj_size, 3), dtype=np.uint8)
        _, video_fr = cap.read()
        video_fr = cv2.resize(video_fr, (proj_size//2, proj_size//2))
        projector[int(y1):int(y2), int(x1):int(x2):] = video_fr
        cv2.imshow('show1', projector)
        cv2.imshow('show2', frame)
        k = cv2.waitKey(20)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    except cv2.error as e:
        print('distance error')

cam.release()
cap.release()
cv2.destroyAllWindows()
