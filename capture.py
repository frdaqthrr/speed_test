import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import math
import pytube
from math import dist
from datetime import datetime

model = YOLO('yolov8s.pt')

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('https://youtu.be/MNn9qKG2UFI?si=EnNo8fJTzbwScjS1')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0

tracker = Tracker()

cy1 = 275
cy2 = 390

offset = 6

vh_down = {}
counter_d = []

vh_up = {}
counter_u = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1050, 500))

    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []
    #detect car
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    #detect truck     
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'truck' in c:
            list.append([x1, y1, x2, y2])
        
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        
        #going down#
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time_d = time.time() - vh_down[id]
                if counter_d.count(id) == 0:
                    counter_d.append(id)
                    distance_d = 10  # meters
                    d_speed_ms = distance_d / elapsed_time_d
                    d_speed_kh = d_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(d_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

                    # Detect kecepatan
                    if d_speed_kh > 60:
                        current_time = datetime.now()
                        timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                        save_directory = "/Users/faridaqthar/yolov8/yolov8_test/gambar/"
                        screenshot_filename = f"speeding_{id}_{timestamp_str}.png"
                        cv2.imwrite(screenshot_filename, frame)

        #going up#
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed_time_u = time.time() - vh_up[id]
                if counter_u.count(id) == 0:
                    counter_u.append(id)
                    distance_u = 10  # meters
                    u_speed_ms = distance_u / elapsed_time_u
                    u_speed_kh = u_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(u_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 255, 255), 2)

                    # Detect kecepatan
                    if u_speed_kh > 60:
                        current_time = datetime.now()
                        timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                        save_directory = "/Users/faridaqthar/yolov8/yolov8_test/gambar/"
                        screenshot_filename = f"speeding_{id}_{timestamp_str}.png"
                        cv2.imwrite(screenshot_filename, frame)

    cv2.line(frame,(1,cy1),(1050,cy1),(255,255,255),1)
    cv2.putText(frame,('L1'),(277,320),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.line(frame,(1,cy2),(1050,cy2),(255,255,255),1)
    cv2.putText(frame,('L2'),(182,367),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    d=(len(counter_d))
    u=(len(counter_u))
    cv2.putText(frame,('goingdown: ')+str(d),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,('goingup: ')+str(u),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
    
cap.release()
cv2.destroyAllWindows()