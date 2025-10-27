import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import os
import numpy as np
import torch
from torchvision import transforms, models
from torch import nn
import time 

# ------------------- CONFIG -------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ASSET_PATH = r"C:\Users\23052\Desktop\CLONE TEST\RoboticsC1-PDE-3802\GUI\assets"
MODELS_PATH = r"C:\Users\23052\Desktop\CLONE TEST\RoboticsC1-PDE-3802\models"

CLASSES = ['eraser', 'glueStick', 'mouse', 'paperClip', 'pen', 
           'pencilBox', 'pencilSharpener', 'scissor', 'stapler', 'waterBottle']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet50 model
MODEL_PATH = os.path.join(MODELS_PATH, "best_resNet50_office_classifier.pth")
num_classes = len(CLASSES)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model = model.to(device)
model.eval()

# GUI implementation
class DetectionApp(ctk.CTk):
    def __init__(self):  # Fixed: double underscore
        super().__init__()  # Fixed: double underscore
        self.title("Object Detection Dashboard")
        self.geometry("950x700")
        self.resizable(False, False)

        self.running = False
        self.cap = None
        self.prob_buffer = []
        self.current_image = None  # To keep reference

        # Gradient background
        self.gradient = ctk.CTkCanvas(self, width=950, height=700, highlightthickness=0)
        self.gradient.place(x=0, y=0)
        self.draw_gradient()

        # Main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#0F0F1A")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)

        # UI Components
        self.create_title()
        self.create_input_frame()
        self.create_output_frame()

    # Gradient Colour Effect
    def draw_gradient(self):
        width, height = 950, 700
        for i in range(height):
            r = int(40 + (i / height) * 80)
            g = int(20 + (i / height) * 40)
            b = int(90 + (i / height) * 120)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.gradient.create_line(0, i, width, i, fill=color)

    # Title
    def create_title(self):
        title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        title_frame.pack(pady=(20, 10))

        logo_path = os.path.join(ASSET_PATH, "object.png")
        if os.path.exists(logo_path):
            logo_img = ctk.CTkImage(Image.open(logo_path), size=(40, 40))
            logo_label = ctk.CTkLabel(title_frame, image=logo_img, text="")
            logo_label.image = logo_img
            logo_label.pack(side="left", padx=10)

        title = ctk.CTkLabel(title_frame, text="Object Detection Dashboard",
                             font=ctk.CTkFont(size=26, weight="bold"),
                             text_color="#E2D9FF")
        title.pack(side="left")

        subtitle = ctk.CTkLabel(self.main_frame,
                                text="ResNet50 Object Detection",
                                font=ctk.CTkFont(size=15),
                                text_color="#A8A3C2")
        subtitle.pack(pady=(0, 20))

    # Buttons
    def create_input_frame(self):
        input_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        input_frame.pack(pady=15, padx=20, fill="x")

        input_label = ctk.CTkLabel(input_frame, text="Choose Input",
                                font=ctk.CTkFont(size=17, weight="bold"))
        input_label.pack(pady=(10, 5))

        buttons_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        buttons_frame.pack(pady=(0, 15))

        # Fixed: Use correct asset paths
        try:
            upload_icon = ctk.CTkImage(Image.open(os.path.join(ASSET_PATH, "upload-file.png")), size=(20, 20))
            camera_icon = ctk.CTkImage(Image.open(os.path.join(ASSET_PATH, "camera.png")), size=(20, 20))
            stop_icon   = ctk.CTkImage(Image.open(os.path.join(ASSET_PATH, "off.png")), size=(20, 20))
        except Exception as e:
            print(f"Icon loading error: {e}")
            # Use default icons if custom ones don't exist
            upload_icon = None
            camera_icon = None
            stop_icon = None

        self.upload_button = ctk.CTkButton(
            buttons_frame, text="Upload Image", width=160, fg_color="#99043F", hover_color="#411023",
            command=self.upload_image, image=upload_icon, compound="left"
        )
        self.upload_button.grid(row=0, column=0, padx=10)

        self.camera_button = ctk.CTkButton(
            buttons_frame, text="Start Camera", width=160, fg_color="#99043F", hover_color="#411023",
            command=self.start_camera, image=camera_icon, compound="left"
        )
        self.camera_button.grid(row=0, column=1, padx=10)

        self.stop_camera_button = ctk.CTkButton(
            buttons_frame, text="Stop Camera", width=160, fg_color="#99043F", hover_color="#411023",
            command=self.stop_camera, image=stop_icon, compound="left"
        )
        self.stop_camera_button.grid(row=0, column=2, padx=10)

    # Output 
    def create_output_frame(self):
        output_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        output_frame.pack(pady=15, padx=20, fill="both", expand=True)

        self.output_label = ctk.CTkLabel(output_frame, text="", font=ctk.CTkFont(size=18))
        self.output_label.pack(pady=(10, 5))

        self.image_label = ctk.CTkLabel(output_frame, text="")
        self.image_label.pack(pady=10)

    # Image Upload 
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            try:
                pil_img = Image.open(path).convert("RGB")
                display_img = pil_img.resize((500, 350))
                
                # Convert to CTkImage
                img_tk = ctk.CTkImage(display_img, size=(500, 350))
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk  # Keep reference
                
                self.stop_camera()

                pred_class, confidence = self.predict_pil(pil_img)
                self.output_label.configure(text=f"Prediction: {pred_class} ({confidence:.1f}%)")
                
            except Exception as e:
                print(f"Error loading image: {e}")
                self.output_label.configure(text="Error loading image")

    # Camera 
    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                self.running = False
                return
            threading.Thread(target=self.camera_loop, daemon=True).start()
            print("Camera started")

    def camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                display_img = pil_img.resize((500, 350))
                
                # Convert to CTkImage
                img_tk = ctk.CTkImage(display_img, size=(500, 350))
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk  # Keep reference

                # Prediction
                probs = self.predict_frame(frame_rgb)
                self.prob_buffer.append(probs)
                if len(self.prob_buffer) > 10:
                    self.prob_buffer.pop(0)

                avg_probs = np.mean(self.prob_buffer, axis=0)
                pred_idx = np.argmax(avg_probs)
                pred_class = CLASSES[pred_idx]
                confidence = avg_probs[pred_idx] * 100

                self.output_label.configure(text=f"Prediction: {pred_class} ({confidence:.1f}%)")

            time.sleep(0.1)  # Add small delay to prevent high CPU usage
            
        if self.cap:
            self.cap.release()
        print("Camera stopped")

    def stop_camera(self):
        self.running = False
        self.prob_buffer.clear()
        print("Stopping camera...")

    # Prediction 
    def predict_pil(self, pil_img):
        try:
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return CLASSES[pred_idx], probs[pred_idx] * 100
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0

    def predict_frame(self, frame_array):
        try:
            pil_img = Image.fromarray(frame_array).convert("RGB")
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"Frame prediction error: {e}")
            return np.zeros(len(CLASSES))

# RUN 
if __name__ == "__main__": 
    print("Starting application...")
    app = DetectionApp()
    app.mainloop()