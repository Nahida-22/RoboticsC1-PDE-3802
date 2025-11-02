# gui.py — ResNet50 Office Items GUI 

import os
from pathlib import Path
import time
import threading

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import json

#  CONFIG 
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Folder that contains GUI asset images.
# Automatically resolve paths relative to this file
BASE_DIR = Path(__file__).resolve().parent

# Folder that contains GUI asset images
ASSET_PATH = BASE_DIR / "assets"


# Models folder
MODELS_PATH = BASE_DIR.parent / "models" 

IMG_SIZE = 224

#  DEVICE 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#  CLASSES 
# Prefer classes.json saved during training (keeps label order exact)
classes_json = MODELS_PATH / "classes.json"
if classes_json.exists():
    try:
        CLASSES = json.loads(classes_json.read_text())
        print(f"Loaded classes from {classes_json}: {CLASSES}")
    except Exception as e:
        print(f"Warning: failed to read {classes_json}: {e}")
        CLASSES = ['eraser', 'glueStick', 'mouse', 'paperClip', 'pen',
                   'pencilBox', 'pencilSharpener', 'scissor', 'stapler', 'waterBottle']
else:
    CLASSES = ['eraser', 'glueStick', 'mouse', 'paperClip', 'pen',
               'pencilBox', 'pencilSharpener', 'scissor', 'stapler', 'waterBottle']

NUM_CLASSES = len(CLASSES)

#  PREPROCESS (match training eval) 
# Training used Resize(1.15*IMG_SIZE) -> CenterCrop(IMG_SIZE) -> Normalize(IMAGENET)
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#  MODEL LOADING 
BEST_PTH = MODELS_PATH / "best.pth"
SCRIPTED_PTH = MODELS_PATH / "model_scripted.pt"
TEMP_PTH = MODELS_PATH / "temperature.pt"

TEMPERATURE = 1.0  # default
if TEMP_PTH.exists():
    try:
        TEMPERATURE = float(torch.load(TEMP_PTH, map_location="cpu").get("temperature", 1.0))
        print(f"Loaded temperature: {TEMPERATURE:.3f}")
    except Exception as e:
        print(f"Warning: failed to load temperature: {e}")

def softmax_with_temp(logits: torch.Tensor, t: float) -> torch.Tensor:
    # Why: calibrate confidence; identical to softmax if t==1
    return torch.softmax(logits / max(t, 1e-6), dim=1)

def build_model(num_classes: int):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model() -> nn.Module:
    # 1) Prefer TorchScript if available (fast + robust)
    if SCRIPTED_PTH.exists():
        try:
            print(f"Loading TorchScript model: {SCRIPTED_PTH}")
            m = torch.jit.load(str(SCRIPTED_PTH), map_location=device)
            m.eval()
            return m
        except Exception as e:
            print(f"Warning: TorchScript load failed ({e}); falling back to state_dict.")

    # 2) Fallback to state_dict checkpoint
    if not BEST_PTH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {BEST_PTH}")

    print(f"Loading checkpoint: {BEST_PTH}")
    ckpt = torch.load(BEST_PTH, map_location=device)

    # Accept multiple formats: {"model": state_dict}, {"model_state_dict": ...}, or raw state_dict
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model", "model_state_dict", "state_dict")):
        state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    else:
        state = ckpt  # assume raw state dict

    m = build_model(NUM_CLASSES).to(device)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict warnings -> missing: {missing}, unexpected: {unexpected}")
    m.eval()
    print("Model loaded successfully!")
    return m

model = load_model()

#  OBJECT DETECTOR (uploaded images only) 
class ObjectDetector:
    """Contour-based detector + classifier for uploaded images only."""

    def __init__(self):
        self.method = 'contour'

    def detect_contour_based(self, image, conf_threshold=0.6):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = image.width * image.height
        min_area = img_area * 0.01
        max_area = img_area * 0.7

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue

            padding = 15
            x1 = max(0, x - padding); y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding); y2 = min(image.height, y + h + padding)

            region = image.crop((x1, y1, x2, y2))
            pred_class, confidence = self.classify_region(region)

            if confidence >= conf_threshold * 100:  # threshold in percentage scale
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': pred_class,
                    'confidence': confidence,
                    'area': area
                })

        detections = sorted(detections, key=lambda d: (d['confidence'], d['area']), reverse=True)
        detections = self.non_max_suppression(detections, iou_threshold=0.3)
        return detections[:5]

    def classify_region(self, region_img):
        try:
            img_tensor = preprocess(region_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = softmax_with_temp(logits, TEMPERATURE).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            return CLASSES[pred_idx], float(probs[pred_idx] * 100.0)
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", 0.0

    def non_max_suppression(self, detections, iou_threshold=0.3):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self.iou(best['bbox'], d['bbox']) < iou_threshold]
        return keep

    def iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min, yi_min = max(x1_min, x2_min), max(y1_min, y2_min)
        xi_max, yi_max = min(x1_max, x2_max), min(y1_max, y2_max)
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - inter_area
        return inter_area / union if union > 0 else 0.0

#  GUI 
class DetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection Dashboard")
        self.geometry("950x700")
        self.resizable(False, False)

        self.running = False
        self.cap = None
        self.prob_buffer = []
        self.current_image = None
        self.detector = ObjectDetector()
        self.show_detections = True

        # Background gradient canvas
        self.gradient = ctk.CTkCanvas(self, width=950, height=700, highlightthickness=0)
        self.gradient.place(x=0, y=0)
        self.draw_gradient()

        # Main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#0F0F1A")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)

        self.create_title()
        self.create_settings_frame()
        self.create_input_frame()
        self.create_output_frame()

    def draw_gradient(self):
        width, height = 950, 700
        for i in range(height):
            r = int(40 + (i / height) * 80)
            g = int(20 + (i / height) * 40)
            b = int(90 + (i / height) * 120)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.gradient.create_line(0, i, width, i, fill=color)

    def create_title(self):
        title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        title_frame.pack(pady=(20, 10))

        logo_path = Path(ASSET_PATH) / "object.png"
        if logo_path.exists():
            logo_img = ctk.CTkImage(Image.open(logo_path), size=(40, 40))
            logo_label = ctk.CTkLabel(title_frame, image=logo_img, text="")
            logo_label.image = logo_img
            logo_label.pack(side="left", padx=10)

        title = ctk.CTkLabel(title_frame, text="Object Detection Dashboard",
                             font=ctk.CTkFont(size=26, weight="bold"),
                             text_color="#E2D9FF")
        title.pack(side="left")

        subtitle = ctk.CTkLabel(self.main_frame,
                                text="ResNet50 Classification",
                                font=ctk.CTkFont(size=15),
                                text_color="#A8A3C2")
        subtitle.pack(pady=(0, 10))

    def create_settings_frame(self):
        settings_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        settings_frame.pack(pady=10, padx=20, fill="x")

        settings_label = ctk.CTkLabel(settings_frame, text="Detection Settings (uploaded images only)",
                                      font=ctk.CTkFont(size=15, weight="bold"))
        settings_label.pack(pady=(10, 5))

        controls_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        controls_frame.pack(pady=(0, 10))

        self.detection_toggle = ctk.CTkCheckBox(
            controls_frame, text="Show Object Detection", command=self.toggle_detections
        )
        self.detection_toggle.select()
        self.detection_toggle.grid(row=0, column=0, padx=20)

        threshold_label = ctk.CTkLabel(controls_frame, text="Confidence:")
        threshold_label.grid(row=0, column=1, padx=5)

        self.threshold_slider = ctk.CTkSlider(controls_frame, from_=0.5, to=0.95, number_of_steps=9,
                                              command=self.update_threshold_label)
        self.threshold_slider.set(0.75)
        self.threshold_slider.grid(row=0, column=2, padx=5)

        self.threshold_value = ctk.CTkLabel(controls_frame, text="75%")
        self.threshold_value.grid(row=0, column=3, padx=5)

    def toggle_detections(self):
        self.show_detections = self.detection_toggle.get()

    def update_threshold_label(self, value):
        self.threshold_value.configure(text=f"{int(float(value)*100)}%")

    def create_input_frame(self):
        input_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        input_frame.pack(pady=10, padx=20, fill="x")

        input_label = ctk.CTkLabel(input_frame, text="Choose Input",
                                   font=ctk.CTkFont(size=17, weight="bold"))
        input_label.pack(pady=(10, 5))

        buttons_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        buttons_frame.pack(pady=(0, 15))

        def load_icon(name):
            p = Path(ASSET_PATH) / name
            if p.exists():
                return ctk.CTkImage(Image.open(p), size=(20, 20))
            return None

        upload_icon = load_icon("upload-file.png")
        camera_icon = load_icon("camera.png")
        stop_icon   = load_icon("off.png")

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

    def create_output_frame(self):
        output_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color="#171729")
        output_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.output_label = ctk.CTkLabel(output_frame, text="", font=ctk.CTkFont(size=18))
        self.output_label.pack(pady=(10, 5))

        self.image_label = ctk.CTkLabel(output_frame, text="")
        self.image_label.pack(pady=10)

    def draw_detections(self, image, detections):
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52C93F']
        for i, det in enumerate(detections):
            bbox = det['bbox']
            label = f"{det['class']}: {det['confidence']:.1f}%"
            color = colors[i % len(colors)]
            draw.rectangle(bbox, outline=color, width=3)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
            draw.rectangle([bbox[0], bbox[1]-25, text_bbox[2]+10, bbox[1]], fill=color)
            draw.text((bbox[0]+5, bbox[1]-22), label, fill='white', font=font)
        return draw_img

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp *.webp")])
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGB")
            self.stop_camera()
            if self.show_detections:
                conf_threshold = self.threshold_slider.get()
                detections = self.detector.detect_contour_based(pil_img, conf_threshold=conf_threshold)
                display_img = self.draw_detections(pil_img, detections).resize((500, 350))
                if detections:
                    det_text = "Found " + ", ".join([f"{d['class']} ({d['confidence']:.1f}%)" for d in detections])
                    self.output_label.configure(text=det_text)
                else:
                    self.output_label.configure(text="No objects detected")
            else:
                display_img = pil_img.resize((500, 350))
                pred_class, confidence = self.detector.classify_region(pil_img)
                self.output_label.configure(text=f"Prediction: {pred_class} ({confidence:.1f}%)")

            img_tk = ctk.CTkImage(display_img, size=(500, 350))
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            print(f"Error loading image: {e}")
            self.output_label.configure(text=f"Error: {e}")

    def start_camera(self):
        if self.running:
            return
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
                img_tk = ctk.CTkImage(display_img, size=(500, 350))
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk

                # Smoothing over last N frames
                probs = self.predict_frame(frame_rgb)
                self.prob_buffer.append(probs)
                if len(self.prob_buffer) > 10:
                    self.prob_buffer.pop(0)
                avg_probs = np.mean(self.prob_buffer, axis=0)
                pred_idx = int(np.argmax(avg_probs))
                pred_class = CLASSES[pred_idx]
                confidence = float(avg_probs[pred_idx] * 100.0)
                self.output_label.configure(text=f"Prediction: {pred_class} ({confidence:.1f}%)")
            time.sleep(0.1)
        if self.cap:
            self.cap.release()
        print("Camera stopped")

    def stop_camera(self):
        self.running = False
        self.prob_buffer.clear()

    def predict_frame(self, frame_array):
        try:
            pil_img = Image.fromarray(frame_array).convert("RGB")
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = softmax_with_temp(logits, TEMPERATURE).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"Frame prediction error: {e}")
            return np.zeros(len(CLASSES), dtype=np.float32)

# RUN
if __name__ == "__main__":
    # Sanity print to confirm the expected files:
    print(f"Looking for model files in: {MODELS_PATH.resolve()}")
    print(f"Exists best.pth? {BEST_PTH.exists()}  | TorchScript? {SCRIPTED_PTH.exists()}  | temperature.pt? {TEMP_PTH.exists()}")

    print("Starting application...")
    app = DetectionApp()
    app.mainloop()