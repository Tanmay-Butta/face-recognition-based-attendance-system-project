import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox as mess
from tkinter import simpledialog
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
from tkinter import ttk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("1100x700")
        
        # We need a shared camera object and a lock so we don't access it concurrently
        self.cam = None
        self.cam_lock = threading.Lock()
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        for F in (AttendancePage, RegistrationPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame("AttendancePage")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            frame.on_show()

    def on_closing(self):
        if self.cam is not None:
            with self.cam_lock:
                self.cam.release()
        self.destroy()

class AttendancePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # UI Layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        
        # Header
        header = ctk.CTkLabel(self, text="Attendance Mode", font=ctk.CTkFont(size=28, weight="bold"))
        header.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Left Panel (Camera)
        left_panel = ctk.CTkFrame(self, fg_color="transparent")
        left_panel.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        self.lbl_video = ctk.CTkLabel(left_panel, text="")
        self.lbl_video.pack(pady=10)
        
        self.btn_mark = ctk.CTkButton(left_panel, text="Press Enter or Click Here to Mark Attendance", font=ctk.CTkFont(size=16), height=50, command=self.mark_attendance)
        self.btn_mark.pack(pady=20, fill="x")
        
        self.lbl_status = ctk.CTkLabel(left_panel, text="Status: Waiting for input...", font=ctk.CTkFont(size=16), text_color="#f1c40f")
        self.lbl_status.pack(pady=10)
        
        # Right Panel (Log)
        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
        log_header = ctk.CTkLabel(right_panel, text="Today's Attendance Log", font=ctk.CTkFont(size=20, weight="bold"))
        log_header.pack(pady=10)
        
        self.tree = ttk.Treeview(right_panel, columns=('id', 'name', 'date', 'time'), show='headings')
        self.tree.heading('id', text='ID')
        self.tree.heading('name', text='NAME')
        self.tree.heading('date', text='DATE')
        self.tree.heading('time', text='TIME')
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        btn_register = ctk.CTkButton(right_panel, text="Manual Registration", fg_color="#e67e22", hover_color="#d35400", command=self.go_to_registration)
        btn_register.pack(pady=20)
        
        self.active = False
        self.check_requested = False
        
        # Load recognizer and dataframe
        self.base_dir = os.path.dirname(__file__)
        self.recognizer = None
        self.faceCascade = None
        self.df = None
        self.load_models()
        
        # Bind Enter Key
        self.controller.bind('<Return>', lambda e: self.mark_attendance() if self.active else None)

    def go_to_registration(self):
        self.active = False
        self.controller.show_frame("RegistrationPage")

    def load_models(self):
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            self.recognizer = cv2.face_LBPHFaceRecognizer.create()
            
        yml_path = os.path.join(self.base_dir, "TrainingImageLabel", "Trainner.yml")
        if os.path.isfile(yml_path):
            self.recognizer.read(yml_path)
            
        csv_path = os.path.join(self.base_dir, "StudentDetails", "StudentDetails.csv")
        if os.path.isfile(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
            
        harcascadePath = os.path.join(self.base_dir, "haarcascade_frontalface_default.xml")
        self.faceCascade = cv2.CascadeClassifier(harcascadePath)

    def on_show(self):
        self.active = True
        self.load_models()
        self.update_log()
        
        # Start camera async to prevent UI lag
        if self.controller.cam is None or not self.controller.cam.isOpened():
            self.lbl_status.configure(text="Status: Initializing Camera... Please wait.", text_color="#e67e22")
            threading.Thread(target=self.init_camera, daemon=True).start()
        else:
            self.lbl_status.configure(text="Status: Waiting for input...", text_color="#f1c40f")
            self.update_frame()
            
    def init_camera(self):
        cam = cv2.VideoCapture(0)
        with self.controller.cam_lock:
            self.controller.cam = cam
        self.after(0, self.camera_ready)
        
    def camera_ready(self):
        if not self.active: return
        self.lbl_status.configure(text="Status: Waiting for input...", text_color="#f1c40f")
        self.update_frame()
        
    def update_frame(self):
        if not self.active:
            return
            
        with self.controller.cam_lock:
            ret, frame = self.controller.cam.read()
            
        if ret:
            frame = cv2.flip(frame, 1)
            
            if self.faceCascade is not None and self.recognizer is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(gray, 1.2, 5)
                
                # If checking attendance but no face detected
                if self.check_requested and len(faces) == 0:
                    self.lbl_status.configure(text="Status: No face detected in frame! Try again.", text_color="#e74c3c")
                    self.check_requested = False
                    
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
                    name = "Unknown"
                    
                    if self.df is not None:
                        try:
                            serial, conf = self.recognizer.predict(gray[y:y + h, x:x + w])
                            if conf < 75:
                                name = self.df.loc[self.df['SERIAL NO.'] == serial]['NAME'].values[0]
                                ID = self.df.loc[self.df['SERIAL NO.'] == serial]['ID'].values[0]
                                
                                if self.check_requested:
                                    self.log_attendance(ID, name, conf)
                                    self.check_requested = False
                            else:
                                if self.check_requested:
                                    self.prompt_unknown()
                                    self.check_requested = False
                        except Exception:
                            if self.check_requested:
                                self.prompt_unknown()
                                self.check_requested = False
                    else:
                        if self.check_requested:
                            self.lbl_status.configure(text="Status: No profiles found! Please register.", text_color="#e74c3c")
                            self.prompt_unknown()
                            self.check_requested = False
                            
                    # Draw name text below rectangle
                    cv2.putText(frame, str(name), (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
            
        self.after(20, self.update_frame)
        
    def mark_attendance(self):
        if not self.active:
            return
        self.check_requested = True
        self.lbl_status.configure(text="Status: Scanning face...", text_color="#3498db")
        
    def log_attendance(self, ID, name, conf):
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        
        self.assure_path_exists(os.path.join(self.base_dir, "Attendance/"))
        att_path = os.path.join(self.base_dir, "Attendance", f"Attendance_{date}.csv")
        exists = os.path.isfile(att_path)
        
        with open(att_path, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            if not exists:
                writer.writerow(['ID', 'NAME', 'DATE', 'TIME'])
            writer.writerow([ID, name, date, timeStamp])
            
        self.lbl_status.configure(text=f"Status: Marked {name} successfully! ({100-int(conf)}%)", text_color="#2ecc71")
        self.update_log()

    def prompt_unknown(self):
        self.lbl_status.configure(text="Status: Unknown face detected!", text_color="#e74c3c")
        # Ask question without blocking the UI thread completely, but askyesno is modal
        res = mess.askyesno("Unknown Face", "Face not recognized.\nWould you like to register a new profile?")
        if res:
            self.go_to_registration()

    def assure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def update_log(self):
        self.tree.delete(*self.tree.get_children())
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        att_path = os.path.join(self.base_dir, "Attendance", f"Attendance_{date}.csv")
        if os.path.isfile(att_path):
            with open(att_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    if len(row) >= 4:
                        self.tree.insert("", "end", values=(row[0], row[1], row[2], row[3]))


class RegistrationPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.base_dir = os.path.dirname(__file__)
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        header = ctk.CTkLabel(self, text="New Profile Registration", font=ctk.CTkFont(size=28, weight="bold"))
        header.grid(row=0, column=0, pady=20)
        
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.grid(row=1, column=0, padx=50, pady=20, sticky="nsew")
        
        # Step 1: Details
        self.frame_details = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_details.pack(pady=50)
        
        ctk.CTkLabel(self.frame_details, text="Enter Student ID:", font=ctk.CTkFont(size=18)).pack(pady=5)
        self.txt_id = ctk.CTkEntry(self.frame_details, width=300, height=40)
        self.txt_id.pack(pady=10)
        
        ctk.CTkLabel(self.frame_details, text="Enter Full Name:", font=ctk.CTkFont(size=18)).pack(pady=5)
        self.txt_name = ctk.CTkEntry(self.frame_details, width=300, height=40)
        self.txt_name.pack(pady=10)
        
        self.btn_next = ctk.CTkButton(self.frame_details, text="Next (Capture Photos)", font=ctk.CTkFont(size=16), height=50, command=self.start_capture)
        self.btn_next.pack(pady=30)
        
        ctk.CTkButton(self.frame_details, text="Cancel & Go Back", fg_color="gray", hover_color="#555555", command=self.back_to_attendance).pack(pady=10)
        
        # Step 2: Capture
        self.frame_capture = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.lbl_video = ctk.CTkLabel(self.frame_capture, text="")
        self.lbl_video.pack(pady=10)
        
        self.lbl_instruction = ctk.CTkLabel(self.frame_capture, text="Please position your face in the oval", font=ctk.CTkFont(size=22, weight="bold"), text_color="#f1c40f")
        self.lbl_instruction.pack(pady=10)
        
        self.progressbar = ctk.CTkProgressBar(self.frame_capture, width=500)
        self.progressbar.pack(pady=10)
        self.progressbar.set(0)
        
        # State variables
        self.capturing = False
        self.sampleNum = 0
        self.active = False
        self.detector = None

    def on_show(self):
        self.txt_id.delete(0, 'end')
        self.txt_name.delete(0, 'end')
        self.frame_capture.pack_forget()
        self.frame_details.pack(pady=50)
        
    def back_to_attendance(self):
        self.active = False
        self.controller.show_frame("AttendancePage")
        
    def assure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def start_capture(self):
        Id = self.txt_id.get()
        name = self.txt_name.get()
        
        if not Id or not name:
            mess.showerror("Error", "Please enter valid ID and Name")
            return
            
        harcascadePath = os.path.join(self.base_dir, "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(harcascadePath)
        
        self.frame_details.pack_forget()
        self.frame_capture.pack(pady=20)
        
        self.sampleNum = 0
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(0)
        self.capturing = True
        self.active = True
        
        if self.controller.cam is None or not self.controller.cam.isOpened():
            self.lbl_instruction.configure(text="Initializing Camera... Please wait.", text_color="#e67e22")
            threading.Thread(target=self.init_camera, daemon=True).start()
        else:
            self.capture_loop()
            
    def init_camera(self):
        cam = cv2.VideoCapture(0)
        with self.controller.cam_lock:
            self.controller.cam = cam
        self.after(0, self.capture_loop)
        
    def get_instruction(self):
        if self.sampleNum < 20: return "Look Straight"
        elif self.sampleNum < 40: return "Look Left"
        elif self.sampleNum < 60: return "Look Right"
        elif self.sampleNum < 80: return "Look Up"
        else: return "Look Down"

    def capture_loop(self):
        if not self.active:
            return
            
        if self.sampleNum >= 100:
            self.capturing = False
            self.lbl_instruction.configure(text="Capture Complete! Preparing to train...", text_color="#2ecc71")
            
            # Use after to allow UI to update before asking password
            self.after(500, self.ask_password)
            return

        with self.controller.cam_lock:
            ret, frame = self.controller.cam.read()
            
        if ret:
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Draw oval
            h, w = display_frame.shape[:2]
            center = (w//2, h//2)
            axes = (w//4, h//3)
            # Oval color changes as it gets closer to 100
            color = (0, int(255 * (self.sampleNum/100)), int(255 * (1 - self.sampleNum/100)))
            cv2.ellipse(display_frame, center, axes, 0, 0, 360, color, 3)
            
            instruction = self.get_instruction()
            self.lbl_instruction.configure(text=f"{instruction} ({self.sampleNum}/100)")
            self.progressbar.set(self.sampleNum / 100)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, fw, fh) in faces:
                self.assure_path_exists(os.path.join(self.base_dir, "TrainingImage/"))
                cv2.imwrite(os.path.join(self.base_dir, "TrainingImage", f"{self.txt_name.get()}.{self.txt_id.get()}.{self.sampleNum}.jpg"), gray[y:y+fh, x:x+fw])
                self.sampleNum += 1
                break # Only capture one face per frame
                
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
            
        self.after(30, self.capture_loop)
        
    def ask_password(self):
        for attempt in range(3):
            pwd = simpledialog.askstring("Admin Authentication", f"Enter Admin Password to save and train profile:\n(Attempt {attempt+1} of 3)", show="*")
            
            if pwd is None:
                mess.showinfo("Cancelled", "Training aborted by user.")
                self.back_to_attendance()
                return
                
            if pwd == "pass@123":
                self.lbl_instruction.configure(text="Training LBPH Model... Please Wait.", text_color="#e67e22")
                self.progressbar.configure(mode="indeterminate")
                self.progressbar.start()
                # Run training in background thread so UI loading bar spins
                threading.Thread(target=self.train_model, daemon=True).start()
                return
            else:
                if attempt < 2:
                    mess.showerror("Error", f"Incorrect Password! You have {2 - attempt} attempts left.")
                    
        mess.showerror("Error", "Incorrect Password! Training aborted after 3 attempts.")
        self.back_to_attendance()

    def train_model(self):
        try:
            # Write to csv
            columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
            serial = 0
            csv_path = os.path.join(self.base_dir, "StudentDetails", "StudentDetails.csv")
            exists = os.path.isfile(csv_path)
            
            if exists:
                with open(csv_path, 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for _ in reader1:
                        serial = serial + 1
                serial = (serial // 2)
            else:
                with open(csv_path, 'a+') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(columns)
                    serial = 1
                    
            Id = self.txt_id.get()
            name = self.txt_name.get()
            row = [serial, '', Id, '', name]
            with open(csv_path, 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
                
            # Train LBPH
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                recognizer = cv2.face_LBPHFaceRecognizer.create()
                
            self.assure_path_exists(os.path.join(self.base_dir, "TrainingImageLabel/"))
            
            faces, Ids = self.getImagesAndLabels(os.path.join(self.base_dir, "TrainingImage"))
            recognizer.train(faces, np.array(Ids))
            recognizer.save(os.path.join(self.base_dir, "TrainingImageLabel", "Trainner.yml"))
            
            self.after(0, self.training_complete)
        except Exception as e:
            print("Training Error:", e)
            self.after(0, lambda: mess.showerror("Error", f"Failed to train model: {str(e)}"))
            self.after(0, self.back_to_attendance)
            
    def training_complete(self):
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate")
        mess.showinfo("Success", "Profile saved and model trained successfully!\nReturning to Attendance Mode.")
        self.back_to_attendance()
        
    def getImagesAndLabels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(ID)
        return faces, Ids

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()
