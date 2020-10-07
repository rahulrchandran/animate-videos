import cv2
import numpy as np
import glob
import os
 
from os.path import isfile, join

class Cartoonizer:

    def __init__(self):
        self.numDownSamples = 1
        self.numBilateralFilters = 7

    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)
        #img_rgb = cv2.resize(img_rgb, (1366,768))
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(self.numDownSamples):
            img_color = cv2.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(self.numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        # upsample image to original size
        for _ in range(self.numDownSamples):
            img_color = cv2.pyrUp(img_color)
        # convert to grayscale and apply bilateral blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        for _ in range(self.numBilateralFilters):
            img_gray_blur = cv2.bilateralFilter(img_gray, 9, 9, 7)
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_gray_blur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 9, 5)
        # convert back to color so that it can be bit-ANDed with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        #Ensure that img_color and img_edge are the same size, otherwise bitwise_and will not work
        height = min(len(img_color), len(img_edge))
        width = min(len(img_color[0]), len(img_edge[0]))
        img_color = img_color[0:height, 0:width]
        img_edge = img_edge[0:height, 0:width]
        return cv2.bitwise_and(img_color, img_edge)

  

def createVideo():
    img_array = []
    for filename in glob.glob('temp/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

 
    #img_array.sort(key = lambda x: x[3:-4])
    img_array.sort()
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_array)):
        print(img_array[i])
        out.write(img_array[i])
    out.release()

def convert_frames_to_video(pathIn1, pathIn2 ,pathOut,fps):
    frame_array1 = []
    files = [f for f in os.listdir(pathIn1) if isfile(join(pathIn1, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[0:-4]))
 
    for i in range(len(files)):
        filename=pathIn1 + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array1.append(img)

    frame_array2 = []
    files = [f for f in os.listdir(pathIn2) if isfile(join(pathIn2, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[0:-4]))
 
    for i in range(len(files)):
        filename=pathIn2 + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array2.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    print(len(frame_array1))
    print(len(frame_array2))

    j = 0 
    for i in range(len(frame_array1)):
        # writing to a image array
        out.write(frame_array1[i])
        out.write(frame_array1[i])
        out.write(frame_array1[i])
        out.write(frame_array1[i])
        out.write(frame_array1[i])
        out.write(frame_array1[i])
        out.write(frame_array2[i])
        out.write(frame_array2[i])
        out.write(frame_array2[i])
        out.write(frame_array2[i])
        out.write(frame_array2[i])
        out.write(frame_array2[i])
        j = i;
    while j< len(frame_array2):
        out.write(frame_array2[j]);
        j = j + 1
    out.release()

'''
cartoonize an image
tmp_canvas = Cartoonizer()
image = "div.jpg" #File_name will come here
image = tmp_canvas.render(image)
cv2.imwrite('div_out.jpg',image)
cv2.destroyAllWindows()
'''

#entries = os.listdir('before/')
#for entry in entries:
#    print(entry)
#cap= cv2.VideoCapture('./before/'+entry)
'''
# input is a video file -- extract each frame as a jpeg image
cap= cv2.VideoCapture('david.mp4')
i=0
j=0
#tmp_canvas = Cartoonizer()
while(cap.isOpened()):
    ret, frame = cap.read()
    i+=1
    if ret == False:
        break
    #frame = tmp_canvas.render(frame)
    cv2.imwrite('temp/david/'+str(j)+'.jpg',frame)
    j+=1
 
cap.release()
'''


'''
# To read images from folder and convert it to a video
pathIn1 = './temp/tony/'
pathIn2 = './temp/david/'
pathOut = './after/IM.mp4'
fps = 29.97
convert_frames_to_video(pathIn1, pathIn2, pathOut, fps)
#createVideo()
#cv2.destroyAllWindows()

#include<iostream>
'''

cap= cv2.VideoCapture('div_1.mp4')
pathOut = './div_after.mp4'
fps = 25
flag = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, layers = frame.shape
    size = (width,height)
    if flag == 0:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
        flag = 1
    out.write(frame)
out.release()
cv2.destroyAllWindows()
