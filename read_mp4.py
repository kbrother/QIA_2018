import os
import cv2
import numpy as np
import TextDataParser
import pdb
from constants import *

def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

path='/mnt/home/qualcomm/Hackathon2018/QIA-Hackathon 2018/Emotion Recognition/Dataset/videos/'

#read file name from csv file
tdp = TextDataParser.TextDataParser()
tdp.parse_file()
cnt = 0
for filename in tdp.get_file_names():
    
    total_fn = path + filename + '.mp4'
    cap = cv2.VideoCapture(total_fn)
    
    #pdb.set_trace()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #cap.set(3, 320)
    #cap.set(4, 240)

    frame_number = NUM_OF_FRAMES_EXTRACTED
    frame_to_read = length // (frame_number + 1)
    
    for i in range(frame_number):
        
        ret = cap.set(1, frame_to_read * i)
#        ret = cap.set(cv2.CAP_PROP_POS_AVI_RATIO, frame_to_read*(i+1))
        ret, frame = cap.read()

        if not ret:
            print(str(i) + " can not read from video")
        else:
            frame = rescale_frame(frame, 50)
            result = cv2.imwrite(os.path.join("./data/frame/", filename + "-%03d"%i + '.png'), frame)
 
  
    cap.release()
    
    cnt = cnt + 1
    if cnt > 10:
        break;
    #success, image = cap.read()
    

