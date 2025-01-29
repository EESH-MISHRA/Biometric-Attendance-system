import os 
import face_recognition
import cv2
import json
import numpy as np

def make_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_serializable(element) for element in data]
    return data


student_img_list = {}
path = "student"    

with open("data.json",'+a') as data:
    for img_file in os.listdir(path):
        img = cv2.imread(f"student\\{img_file}")
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_encoding = face_recognition.face_encodings(rgb)[0]
        converted_img_array = make_serializable(img_encoding)
        student_img_list[img_file[:-4]] = [img_file[:-4],converted_img_array]
    json.dump(student_img_list,data)


