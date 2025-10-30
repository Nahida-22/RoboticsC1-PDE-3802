import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
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
MODEL_PATH = os.path.join(MODELS_PATH, "best_resNet50_office_classifier_2.pth")
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

# ------------------- OBJECT DETECTION FOR UPLOADED IMAGES -------------------

class ObjectDetector:
    """Handles object detection using contour-based method only for uploaded images"""
    
    def __init__(self):
        self.method = 'contour'
    
    def detect_contour_based(self, image, conf_threshold=0.6):
        """
        Contour-based detection - finds objects based on edges
        Good for objects with clear boundaries
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate image area for filtering
        img_area = image.width * image.height
        min_area = img_area * 0.01  # Minimum 1% of image
        max_area = img_area * 0.7   # Maximum 70% of image
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (reject very thin/wide boxes)
            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Add padding
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding)
            y2 = min(image.height, y + h + padding)
            
            # Classify region
            region = image.crop((x1, y1, x2, y2))
            pred_class, confidence = self.classify_region(region)
            
            if confidence > conf_threshold * 100:  # Convert threshold to percentage
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': pred_class,
                    'confidence': confidence,
                    'area': area
                })
        
        # Sort by confidence and area, keep best detections
        detections = sorted(detections, key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        # Apply Non-Maximum Suppression with stricter threshold
        detections = self.non_max_suppression(detections, iou_threshold=0.3)
        
        # Limit to top 5 detections
        return detections[:5]
    
    def classify_region(self, region_img):
        """Classify a cropped region using ResNet50"""
        try:
            img_tensor = preprocess(region_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return CLASSES[pred_idx], probs[pred_idx] * 100
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", 0.0
    
    def non_max_suppression(self, detections, iou_threshold=0.3):
        """Remove overlapping bounding boxes - stricter by default"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            detections = [
                d for d in detections 
                if self.iou(best['bbox'], d['bbox']) < iou_threshold
            ]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

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
        
        # Object detector for uploaded images only
        self.detector = ObjectDetector()
        self.show_detections = True

        # Gradient background
        self.gradient = ctk.CTkCanvas(self, width=950, height=700, highlightthickness=0)
        self.gradient.place(x=0, y=0)
        self.draw_gradient()

        # Main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#0F0F1A")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)

        # UI Components
        self.create_title()
        self.create_settings_frame()
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
        subtitle.pack(pady=(0, 10))

    # Settings Frame for Uploaded Images
    def create_settings_frame(self):
        settings_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        settings_frame.pack(pady=10, padx=20, fill="x")

        settings_label = ctk.CTkLabel(settings_frame, text="Detection Settings (for uploaded images only)",
                                      font=ctk.CTkFont(size=15, weight="bold"))
        settings_label.pack(pady=(10, 5))

        controls_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        controls_frame.pack(pady=(0, 10))

        # Toggle detection checkbox (only affects uploaded images)
        self.detection_toggle = ctk.CTkCheckBox(
            controls_frame,
            text="Show Object Detection",
            command=self.toggle_detections
        )
        self.detection_toggle.select()
        self.detection_toggle.grid(row=0, column=0, padx=20)

        # Confidence threshold slider (only for uploaded images)
        threshold_label = ctk.CTkLabel(controls_frame, text="Confidence:")
        threshold_label.grid(row=0, column=1, padx=5)

        self.threshold_slider = ctk.CTkSlider(
            controls_frame,
            from_=0.5,
            to=0.95,
            number_of_steps=9
        )
        self.threshold_slider.set(0.75)
        self.threshold_slider.grid(row=0, column=2, padx=5)

        self.threshold_value = ctk.CTkLabel(controls_frame, text="75%")
        self.threshold_value.grid(row=0, column=3, padx=5)

        self.threshold_slider.configure(command=self.update_threshold_label)

    def toggle_detections(self):
        self.show_detections = self.detection_toggle.get()

    def update_threshold_label(self, value):
        self.threshold_value.configure(text=f"{int(float(value)*100)}%")

    # Buttons
    def create_input_frame(self):
        input_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        input_frame.pack(pady=10, padx=20, fill="x")

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
        output_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.output_label = ctk.CTkLabel(output_frame, text="", font=ctk.CTkFont(size=18))
        self.output_label.pack(pady=(10, 5))

        self.image_label = ctk.CTkLabel(output_frame, text="")
        self.image_label.pack(pady=10)

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image - for uploaded images only"""
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52C93F']
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            label = f"{det['class']}: {det['confidence']:.1f}%"
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label background
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle(
                [bbox[0], bbox[1] - 25, text_bbox[2] + 10, bbox[1]],
                fill=color
            )
            draw.text((bbox[0] + 5, bbox[1] - 22), label, fill='white', font=font)
        
        return draw_img

    # Image Upload with Object Detection
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            try:
                pil_img = Image.open(path).convert("RGB")
                self.stop_camera()

                if self.show_detections:
                    # Perform object detection using contour method
                    conf_threshold = self.threshold_slider.get()
                    
                    detections = self.detector.detect_contour_based(
                        pil_img, conf_threshold=conf_threshold
                    )
                    
                    # Draw bounding boxes
                    display_img = self.draw_detections(pil_img, detections)
                    display_img = display_img.resize((500, 350))
                    
                    # Update output label
                    if detections:
                        det_text = f"Found {len(detections)} object(s): " + \
                                  ", ".join([f"{d['class']} ({d['confidence']:.1f}%)" 
                                           for d in detections])
                        self.output_label.configure(text=det_text)
                    else:
                        self.output_label.configure(text="No objects detected")
                else:
                    # Simple classification without bounding boxes
                    display_img = pil_img.resize((500, 350))
                    pred_class, confidence = self.detector.classify_region(pil_img)
                    self.output_label.configure(text=f"Prediction: {pred_class} ({confidence:.1f}%)")
                
                # Display image
                img_tk = ctk.CTkImage(display_img, size=(500, 350))
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk
                
            except Exception as e:
                print(f"Error loading image: {e}")
                self.output_label.configure(text=f"Error: {e}")

    # Camera - EXACTLY THE SAME AS YOUR ORIGINAL CODE
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

    # Prediction - EXACTLY THE SAME AS YOUR ORIGINAL CODE
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