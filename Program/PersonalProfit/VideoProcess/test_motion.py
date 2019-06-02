#!bin/bash/env python
# -*- coding:utf-8 -*-

# 通过计算两帧图像之间变化了的像素点占的百分比，来确定图像中是否有动作产生。

# 这里主要用到 Absdiff 函数，比较两帧图像之间有差异的点，
# 当然需要将图像进行一些处理，例如平滑处理，灰度化处理，二值化处理，
# 经过处理之后的二值图像上的点将更有效。

import cv2.cv as cv

capture = cv.VideoCapture(0)  # 改成视频路径

frame1 = cv.QueryFrame(capture)
frame1gray = cv.CreateMat(frame1.height, frame1.width, cv.CV_8U)
cv.CvtColor(frame1, frame1gray, cv.CV_RGB2GRAY)

res = cv.CreateMat(frame1.height, frame1.width, cv.CV_8U)

frame2gray = cv.CreateMat(frame1.height, frame1.width, cv.CV_8U)

w= frame2gray.width
h= frame2gray.height
nb_pixels = frame2gray.width * frame2gray.height

while True:
    frame2 = cv.QueryFrame(capture)
    cv.CvtColor(frame2, frame2gray, cv.CV_RGB2GRAY)

    cv.AbsDiff(frame1gray, frame2gray, res)
    cv.ShowImage("After AbsDiff", res)

    cv.Smooth(res, res, cv.CV_BLUR, 5,5)
    element = cv.CreateStructuringElementEx(5*2+1, 5*2+1, 5, 5,  cv.CV_SHAPE_RECT)
    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_OPEN)
    cv.MorphologyEx(res, res, None, None, cv.CV_MOP_CLOSE)
    cv.Threshold(res, res, 10, 255, cv.CV_THRESH_BINARY_INV)

    cv.ShowImage("Image", frame2)
    cv.ShowImage("Res", res)

    #-----------
    nb=0
    for y in range(h):
        for x in range(w):
            if res[y,x] == 0.0:
                nb += 1
    avg = (nb*100.0)/nb_pixels
    if avg >= 5:
        print u"有物体移动!"
    #-----------
    cv.Copy(frame2gray, frame1gray)
    c=cv.WaitKey(1)
    if c==27: # 按'Esc'.退出
        break