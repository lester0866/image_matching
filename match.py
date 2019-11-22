import cv2
import pafy
import numpy as np
from tkinter import *
from tkinter import ttk
import pandas as pd

data_loc = '/home/lester/Desktop/data.csv'
df = pd.read_csv(data_loc)
url = None


def get_filenames():
    return df.iloc[:, 0].tolist()


def onselect(evt):
    # Note here that Tkinter passes an event object to onselect()
    w = evt.widget
    index = int(w.curselection()[0])
    global url
    url = df.iloc[index, 1]
    print('You selected item %d: "%s"' % (index, url))
    root.destroy()


root = Tk()
l = Listbox(root, selectmode=SINGLE, height=30, width=60)
l.grid(column=0, row=0, sticky=(N, W, E, S))
s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
s.grid(column=1, row=0, sticky=(N, S))
l['yscrollcommand'] = s.set
ttk.Sizegrip().grid(column=1, row=1, sticky=(S, E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.geometry('350x500+50+50')
root.title('Select Video')
for filename in get_filenames():
    l.insert(END, filename)
l.bind('<<ListboxSelect>>', onselect)
root.mainloop()

proj_size = 1024
cv2.namedWindow("show1")
cv2.namedWindow("show2")
cam = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480

# url = 'https://youtu.be/0usXEzuKdkU'
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
frame_counter = 0
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
while True:
    try:
        _, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(frame, None)
        # Match descriptors.
        if des2 is not None:
            good_matches = bf.match(des1, des2)
            distance_sum = int(sum([m.distance for m in good_matches]))
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            centroid = (dst_pts.sum(0) / len(dst_pts)).squeeze().astype('int')
            frame = cv2.circle(frame, tuple(centroid), 10, (255, 255, 255), -1)
            offset = np.array([cam_w // 2, cam_h // 2]) - centroid
            curr_centroid = np.array([proj_size // 2, proj_size // 2]) - offset
            if np.sum(np.abs(curr_centroid - proj_centroid)) >= 50 and distance_sum >= 3000:
                proj_centroid = curr_centroid
                x1, y1 = proj_centroid - proj_size // 8
                x2, y2 = proj_centroid + proj_size // 8
        projector = np.zeros((proj_size, proj_size, 3), dtype=np.uint8)
        _, video_fr = cap.read()
        video_fr = cv2.resize(video_fr, (proj_size // 4, proj_size // 4))
        projector[int(y1):int(y2), int(x1):int(x2):] = video_fr
        cv2.imshow('show1', projector)
        cv2.imshow('show2', frame)
        k = cv2.waitKey(20)
        frame_counter += 1
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap = cv2.VideoCapture(play.url)
    except cv2.error as e:

        print('distance error')
        print(e)

cam.release()
cap.release()
cv2.destroyAllWindows()
