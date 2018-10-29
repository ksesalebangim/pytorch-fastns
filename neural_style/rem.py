import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
  help = "path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
  cap = cv2.VideoCapture(0)
else:
  cap = cv2.VideoCapture(args["video"])

fgbg = cv2.BackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
