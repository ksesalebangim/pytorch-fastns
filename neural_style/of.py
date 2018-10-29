import cv2
from numpy import *
import numpy as np
from skimage.measure import compare_ssim
def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis


cap = cv2.VideoCapture(0)

ret,im = cap.read()

prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(im)
hsv[...,1] = 255
while True:
    # get grayscale image
    ret,im = cap.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # compute flow
    #flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    (score, diff) = compare_ssim(prev_gray, gray, full=True)
    diff = (diff * 255).astype("uint8")

    mask = cv2.threshold(diff, 0, 255,
         cv2.THRESH_OTSU)[1]
    mask = cv2.medianBlur(mask, 3)
    fg = cv2.bitwise_or(prev_gray, prev_gray, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(gray, gray, mask=mask)
    #bk = self.bgEfect
    final = cv2.bitwise_or(fg, bk)
    final = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    bgr =cv2.bitwise_or(bgr, bgr)
    prev_gray = gray

    # plot the flow vectors
    cv2.imshow('Optical flow',bgr)
    if cv2.waitKey(10) == 27:
        break
