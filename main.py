import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import threading
import sqlite3
import datetime

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("800x600")
        
        # Directories and paths
        self.data_dir = "dataset"
        os.makedirs(self.data_dir, exist_ok=True)
            
        self.model_path = "trainer.yml"
        
        # Load cascade and recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Database setup
        self.db_path = "facedata.db"
        self.setup_database()
        
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            
        # Video Capture state
        self.video_capture = None
        self.is_capturing = False
        
        # Training state
        self.is_saving = False
        self.sample_count = 0
        self.max_samples = 60 # 60 samples is enough
        self.current_name = ""
        self.current_id = 1
        
        # Build GUI
        self.setup_gui()
        self.update_frame_job = None
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def get_user_name(self, id_num):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE id = ?", (id_num,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"

    def get_all_users(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users")
        result = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in result}

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=1, fill="both")
        
        self.setup_train_tab()
        self.setup_recog_tab()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def setup_train_tab(self):
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Data Train")
        
        controls = ttk.Frame(self.train_tab)
        controls.pack(side="top", fill="x", pady=10)
        
        self.btn_start = ttk.Button(controls, text="Start", command=self.start_camera_train)
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_stop = ttk.Button(controls, text="Stop", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(side="left", padx=5)
        
        ttk.Label(controls, text="Person Name:").pack(side="left", padx=5)
        self.name_entry = ttk.Entry(controls)
        self.name_entry.pack(side="left", padx=5)
        
        self.btn_save = ttk.Button(controls, text="Save", command=self.start_saving_faces, state=tk.DISABLED)
        self.btn_save.pack(side="left", padx=5)
        
        self.train_video_label = ttk.Label(self.train_tab)
        self.train_video_label.pack(expand=1, fill="both")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.train_tab, variable=self.progress_var, maximum=self.max_samples)
        self.progress_bar.pack(fill="x", side="bottom", padx=10, pady=10)

    def setup_recog_tab(self):
        self.recog_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recog_tab, text="Recognition")
        
        controls = ttk.Frame(self.recog_tab)
        controls.pack(side="top", fill="x", pady=10)
        
        self.btn_start_recog = ttk.Button(controls, text="Start Recognition", command=self.start_camera_recog)
        self.btn_start_recog.pack(side="left", padx=5)
        
        self.btn_stop_recog = ttk.Button(controls, text="Stop Recognition", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop_recog.pack(side="left", padx=5)
        
        self.recog_video_label = ttk.Label(self.recog_tab)
        self.recog_video_label.pack(expand=1, fill="both")

    def on_tab_changed(self, event):
        self.stop_camera()

    def start_camera_train(self):
        if not self.is_capturing:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.is_capturing = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
            self.update_train_frame()

    def start_camera_recog(self):
        if not self.is_capturing:
            if not os.path.exists(self.model_path):
                messagebox.showerror("Error", "No trained model found. Please train data first.")
                return
            try:
                self.recognizer.read(self.model_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                return
                
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.is_capturing = True
            self.btn_start_recog.config(state=tk.DISABLED)
            self.btn_stop_recog.config(state=tk.NORMAL)
            self.update_recog_frame()

    def stop_camera(self):
        self.is_capturing = False
        if self.update_frame_job:
            self.root.after_cancel(self.update_frame_job)
            self.update_frame_job = None
            
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_start_recog.config(state=tk.NORMAL)
        self.btn_stop_recog.config(state=tk.DISABLED)
        
        self.train_video_label.config(image='')
        self.recog_video_label.config(image='')
        
        self.is_saving = False
        self.progress_var.set(0)

    def start_saving_faces(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a valid name.")
            return
            
        self.current_name = name
        
        # Check if user already exists in DB
        users = self.get_all_users()
        found_id = None
        for id_num, n in users.items():
            if n.lower() == name.lower():
                found_id = id_num
                break
                
        if found_id is not None:
            self.current_id = found_id
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", 
                           (name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            self.current_id = cursor.lastrowid
            conn.close()
            
        self.person_dir = os.path.join(self.data_dir, str(self.current_id))
        os.makedirs(self.person_dir, exist_ok=True)
            
        self.sample_count = 0
        self.is_saving = True
        self.btn_save.config(state=tk.DISABLED)
        self.name_entry.config(state=tk.DISABLED)

    def update_train_frame(self):
        if self.is_capturing and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.flip(frame, 1) # Mirror
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    if self.is_saving:
                        if self.sample_count < self.max_samples:
                            face_img = gray[y:y+h, x:x+w]
                            file_path = os.path.join(self.person_dir, f"{self.current_id}_{self.sample_count}.jpg")
                            cv2.imwrite(file_path, face_img)
                            self.sample_count += 1
                            self.progress_var.set(self.sample_count)
                        else:
                            self.is_saving = False
                            self.progress_var.set(0)
                            self.btn_save.config(state=tk.NORMAL)
                            self.name_entry.config(state=tk.NORMAL)
                            
                            # Train model in background thread
                            threading.Thread(target=self.train_model, daemon=True).start()
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.train_video_label.imgtk = imgtk
                self.train_video_label.configure(image=imgtk)
            
            self.update_frame_job = self.root.after(15, self.update_train_frame)

    def train_model(self):
        self.root.after(0, lambda: self.btn_save.config(state=tk.DISABLED, text="Training..."))
        
        faces = []
        ids = []
        
        for root_dir, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("png"):
                    path = os.path.join(root_dir, file)
                    try:
                        id_num = int(os.path.basename(root_dir))
                        img = Image.open(path).convert('L')
                        image_np = np.array(img, 'uint8')
                        faces.append(image_np)
                        ids.append(id_num)
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
                        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.model_path)
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Data for '{self.current_name}' saved and recognized model updated!"))
        else:
            self.root.after(0, lambda: messagebox.showwarning("Training Error", "No suitable training data found."))
            
        self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL, text="Save"))

    def update_recog_frame(self):
        if self.is_capturing and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.flip(frame, 1) # Mirror
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    id_num, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 80: # Lower distance means higher confidence
                        name = self.get_user_name(id_num)
                        color = (0, 255, 0) # Green square for recognized
                    else:
                        name = "Unknown"
                        color = (0, 0, 255) # Red square for unknown
                        
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.recog_video_label.imgtk = imgtk
                self.recog_video_label.configure(image=imgtk)
            
            self.update_frame_job = self.root.after(15, self.update_recog_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()
