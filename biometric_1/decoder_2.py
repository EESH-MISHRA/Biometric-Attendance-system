
import json
import numpy as np


def load_encodings():
    '''decoder file execution and img list to array conversion'''
    with open("data.json", 'r') as data_file:
        student_img_list = json.load(data_file)
        for key, value in student_img_list.items():
            student_img_list[key][1] = np.array(value[1]) 
    return student_img_list
# Prepare the list of known face encodings and corresponding names
def prepare_known_faces(student_img_list):
    known_face_encodings = []
    known_face_names = []

    for name, value in student_img_list.items():
        encoding = value[1]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

    return known_face_encodings, known_face_names
if __name__ =="__main__":
    load_encodings()
    prepare_known_faces()
       





