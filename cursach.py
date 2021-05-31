import random
import numpy as np 
import math
from cv2 import cv2

try:  
    original = cv2.imread("test.jpg")
except FileNotFoundError:  
    print("Файл не найден")

def angle(point_1, point_2, point_3):
    line1_x = point_1[0] - point_2[0]
    line2_x = point_3[0] - point_2[0]
    line1_y = point_1[1] - point_2[1]
    line2_y = point_3[1] - point_2[1]
    numerator = line1_x*line2_x + line1_y*line2_y
    denominator = math.sqrt(line1_x**2+line1_y**2)*math.sqrt(line2_x**2+line2_y**2)
    return numerator/denominator

def lomanaya(min_delta, max_delta, length,now_x,now_y):
    res = [(now_x, now_y)]
    for _ in range(length):
        new_x = random.uniform(min_delta, max_delta) * random.choice((-1,1))
        new_y = random.uniform(min_delta, max_delta) * random.choice((-1,1))
        if len(res)<3:
            now_x += new_x
            now_y += new_y
        else:
            while not (-1 <= angle(res[-2],res[-1],(now_x+new_x,now_y+new_y)) <= 0.866): # and math.sqrt(1-angle(res[-2],res[-1],(now_x+new_x,now_y+new_y))**2)>0
                new_x = random.uniform(min_delta, max_delta) * random.choice((-1,1))
                new_y = random.uniform(min_delta, max_delta) * random.choice((-1,1)) 
            now_x += new_x
            now_y += new_y
        res.append((int(now_x), int(now_y)))
    return res

#im = Image.new('RGB', (1000, 500), (128, 128, 128))
w = original.shape[0] // 2
h = original.shape[1] // 2
n = 10
max_step = 100
points = lomanaya(10,max_step,n,w,h)
black_img = np.zeros((original.shape[0],original.shape[1],3), np.uint8)
true_value_img = np.zeros((original.shape[0],original.shape[1],3), np.uint8)

for i in range(len(points)-1):
    width = random.choice(range(1,7))
    cv2.line(black_img,points[i],points[i+1],(128,128,128),thickness=width)
    cv2.line(true_value_img,points[i],points[i+1],(255,255,255),thickness=width)
crack_image = cv2.subtract(original,black_img)
Hori = np.concatenate((crack_image, true_value_img), axis=1)
cv2.imshow('crack', Hori)
cv2.waitKey(0)
