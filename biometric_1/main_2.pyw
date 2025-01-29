import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.neighbors import NearestNeighbors
import threading
from decoder_2 import load_encodings,prepare_known_faces



def mark_attendance(name, attendance_df):
    '''create dataframe to mark attandence'''
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # check condition it attendance is marked on the same date 
    if not ((attendance_df["Name"] == name) & (attendance_df["Date"] == date)).any():
        new_row = pd.DataFrame({"Name": [name], "Date": [date], "Time": [time]})
        attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
        # save todays data in excel file
        attendance_df.to_excel(f"attendance_log_{date}.xlsx", index=False)
        return attendance_df, date, time
    return attendance_df, None, None


def show_popup(name, date, time):
    '''show pop-up message for attendance'''
    messagebox.showinfo("Attendance Marked", f"Attendance marked for {name} on {date} at {time}")

from customtkinter import CTkImage
from PIL import Image, ImageTk

def video_feed(video_label, known_face_encodings, known_face_names, student_img_list, attendance_df):
    web_cam = cv2.VideoCapture(0)
    frame_count = 0
    neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    neigh.fit(known_face_encodings)

    while True:
        ret, video_frame = web_cam.read()
        if not ret:
            break

        # Resize the video frame for lower quality and smooth efficiency
        small_frame = cv2.resize(video_frame, (0, 0), fx=0.5, fy=0.5)

        frame_count += 1
        if frame_count % 2 != 0:  # skips every next frame for faster processing
            continue

        # Detect faces
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # performing nearest-neighbour search for faster processing
            distances, indices = neigh.kneighbors([face_encoding])

            if distances[0][0] < 0.6:  # Threshold for face match
                matched_name = known_face_names[indices[0][0]]
                attendance_df, date, time = mark_attendance(matched_name, attendance_df)
                if date and time:
                    show_popup(matched_name, date, time)

        # Display the video frame in the Tkinter window
        cv2image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img_tk = ImageTk.PhotoImage(img)
        video_label.configure(image=img_tk)
        video_label.image = img_tk  # Keep a reference to avoid garbage

        app.update()



# tkinter window
app = ctk.CTk()
app.iconbitmap("icon.ico")
app.geometry("1200x600")
app.title("Attendance System")

# attendance dataframe format
attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])

# layout for video frame
frame = ctk.CTkFrame(app)
frame.pack(fill="both", expand=True)

video_label = ctk.CTkLabel(frame,text="")
video_label.pack(side="right", fill="both", expand=True)
# side layout
left_frame = ctk.CTkFrame(frame, width=300)
left_frame.pack(side="left", fill="y")

label = ctk.CTkLabel(left_frame, text="Attendance System\nMade By\nEesh Mishra", font=("Arial", 20))
label.pack(padx=20, pady=20)

# load prepare faces from decoder.py
student_img_list = load_encodings()
known_face_encodings, known_face_names = prepare_known_faces(student_img_list)

# Start the video feed in a separate thread to prevent blocking
video_thread = threading.Thread(target=video_feed, args=(video_label, known_face_encodings, known_face_names, student_img_list, attendance_df))
video_thread.daemon = True
video_thread.start()

app.mainloop()
