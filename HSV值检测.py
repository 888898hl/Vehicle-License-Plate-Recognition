import cv2
import numpy as np
from matplotlib import pyplot as plt
def plt_show0(img):#plt显示彩色图
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
picture=cv2.imread("picture.jpg")#car6,7
picture=cv2.resize(picture,(480,190))
#image=cv2.imread('szy.png')这是直接标定图片
HSV=cv2.cvtColor(picture,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
        print(HSV[y,x])

cv2.imshow("imageHSV",HSV)
cv2.imshow('image',picture)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey(0)


