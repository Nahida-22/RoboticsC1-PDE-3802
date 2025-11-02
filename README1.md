# Office Classifier Items

This readme file gives a whole overview of how this project was done, starting with the images collection, data cleaning, augmentation and data splitting for training (CNN and Transfer Learning). Evaluation metrics were also displayed for each model training. We also have a GUI for file upload and live camera, which guesses what is the object being shown.


# Running Instructions (Python 3.10)

1. Install dependencies: 

2. If you only want to run the GUI, you don’t need TensorFlow or seaborn/pandas.
    pip install --upgrade pip
    pip install customtkinter pillow opencv-python numpy tqdm scikit-learn
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    # (If you have CUDA or Apple Silicon MPS, use the install selector on pytorch.org instead)

If you will run notebook, you will find the required librairies to install documented there.

3. Ensure project structure & model files
        RoboticsC1-PDE-3802/
        ├─ GUI/
        │  ├─ assets/
        │  │  ├─ object.png
        │  │  ├─ camera.png
        │  │  ├─ off.png
        │  │  └─ upload-file.png
        │  └─ gui.py
        └─ Notebooks/
        └─ models/
            ├─ best.pth               
                    
4. Configure paths (if needed)

5. Run the GUI
    cd GUI
    python gui.py

6. Notes & troubleshooting

    PyTorch build: If you have CUDA or Apple Silicon (MPS), install PyTorch using the command from pytorch.org
    for your hardware.

    Camera permissions (macOS): System Settings → Privacy & Security → Camera → allow Terminal/Python.

    Model not found: On startup, the app prints which files it found in outputs_resnet50_clean. Ensure at least best.pth or model_scripted.pt exists.

    Fonts: If arial.ttf isn’t available, the GUI falls back to the default font automatically.

    Python version: This project was developed on Python 3.10; use 3.10.x for best compatibility.




# Library installation 

Ensure you have Python 3.10.11, this is what was used as the virtual environmment.

    ```bash
    # Create virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\Scripts\activate         # Windows

    # Install core dependencies
    pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm tensorflow
    pip install torch torchvision torchaudio
    pip install jupyter notebook ultralytics
    ``` 
# DATASET CREATION

After installing all librairies, we will go through our dataset creation:

    1. Images were collected from the internet through various sites
    2. Those images were assigned a specific class ("Pen", "Water Bottle", "Stapler", etc..)
    4. The number of images in each class folder was checked to determine if data augmentation  was needed and to identify classes with fewer images.
    5. Since some images have been taken from the internet, and others were captured from mobile phone camera, the images are of different sizes. Therefore, resizing is important before training a machine learning model as it makes the dataset uniform, memory-efficient, and compatible with the model architecture, which is crucial for effective training.
    6. 224×224 is chosen because it is a good compromise between preserving object detail and computational efficiency, and it matches the input size of most pre-trained models used for transfer learning, and then saved to "processed" folder
    7. It can be deduced that there is a class imbalance. Some classes may have fewer images than others, indicating that data augmentation or additional data collection is needed to balance the dataset. Classes with a low number of images  for e.g paper clips, are candidates for augmentation to increase training data and improve model performance.
    8. For augmentation, rotate, flip and change hue was used to create more images for imbalanced classes, and then saved to "augmented" folder.
    9. After using augmentation, each image was renamed depending on the class they belong to, for e.g. pen0001.jpg, etc... 
    10. Then, the dataset (augmented) was splitted. (train=80%, test=15% and val=5%)

# IMAGES PER CLASSES BEFORE AUGMENTATION

Below you see the number of images per class before doing augmentation:
Class: waterBottle, Number of images: 1195
Class: pen, Number of images: 2282
Class: mouse, Number of images: 1693
Class: eraser, Number of images: 452
Class: glueStick, Number of images: 412
Class: scissor, Number of images: 812
Class: stapler, Number of images: 311
Class: pencilBox, Number of images: 487
Class: pencilSharpener, Number of images: 587
Class: paperClip, Number of images: 332

# DATA AUGMENTATION

Data augmentation techniques were applied to improve generalization:
- Rotation
- Flipping (horizontal/vertical)
- Changing hue

Example augmentation code snippet:

```python

def rotate(image):
    angle = random.choice([15, -15, 30, -30])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip(image):
    return cv2.flip(image, 1)

def change_hue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(-20, 20)
    hsv_h = hsv[:, :, 0].astype(int)
    hsv[:, :, 0] = ((hsv_h + hue_shift) % 180).astype('uint8')
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

augmentations = [rotate, flip, change_hue]
```

Total images after augmentation: 30000
Total classes: 10

# DATASET FOLDER STRUCTURE

```
dataset/
├── train/
│   ├── pen/
│   ├── stapler/
│   └── tape/
├── val/
│   ├── pen/
│   ├── stapler/
│   └── tape/
└── test/
    ├── pen/
    ├── stapler/
    └── tape/
```

# TRAINING THE DATASET WITH CNN

The model consisted of two convolutional layers followed by max pooling, dropout and fully connected layers. It was trained on the augmentes dataset for 20 epochs using the Adam Optimizer and cross-entropy loss.

During testing, the model achieved moderate performance, as it successfully learned basic shapes and texture features, but struggled with more sudden variations in lightning and orientation.

Accuracy: 85.2%
Precision: 84%
Recall: 85%
F1-Score: 84.6%

The observations that were made is shown below:

    1. Confusion matrix shows misclassifications between visually similar items
    2. The model converges quickly but plateaus after 15 epochs, which suggests limited feature extraction capacity
    3. Overfitting was controlled by dropout but the val accuracy remained lower than desired
    4. After seeing those results, we decided to go for Transfer Learning and look for the perfect model for our task

ResNet50 was selected because it delivers high accuracy with stable training dynamics, efficient computation, and excellent feature extraction — ideal for detecting and classifying small, visually similar office items.



# TRAINING USING RESNET-50 ON A SMALL DATASET (ALL LAYERS FROZEN)

ResNet-50 was chosen for its strong feature extraction capabilities, efficient computation, and proven transfer performance.

For this experiment, all convolutional layers were frozen, and only the final fully connected layer was trained for 5 epochs on a small subset of the full dataset.

Images were resized to 224×224 pixels and normalized using ImageNet statistics.

During training, the model achieved stable learning but began to overfit after the fourth epoch, as shown by a drop in validation accuracy and F1-score.

Most classes were predicted correctly, with minor cross-class confusion visible in the confusion matrix.

# 2ND TRAINING WITH 4TH LAYER UNFROZE

For this 2nd training, we chose to unfreeze the 4th layer and the FC head, allowing the model to adapt high level features, the preprocessing remains the same.

    Accuracy: 88%
    Macro-F1: 0.879

Fine-tuning ResNet-50’s last block (layer4) improved validation performance from 82%/0.825 (frozen) to 88%/0.879, reducing cross-class confusions and confirming the benefit of adapting high-level features on the small subset.

# 1st training on the main dataset (30000 images)

In this training, we tried to reduce overfitting seen in the previous training (small dataset). We unfroze the 4th and 3rd layer and the fully connected head, while keeping earlier layers frozen to preserve low-level features representations.

We used Adam with different learning rates (1e-5 for layer 3 and 4, and 1e-4 for the FC head), cross-entropy loss, a ReduceLROnPlateau scheduler which monitors val F1 and also early stopping on 5 consecutive epochs if no improvement is made. The model was trained for 20 epochs and batch size 32.

    Accuracy: 99.1%
    Precision: 0.99
    Recall: 0.99
    Confusion matrix: Only minor off-class errors

Unfreezing deeper layers led to a major jump in performance, from 82-88 % accuracy to 99% on our full dataset. We also saved the best epoch model "best_resNet50_office_classifier.pth", which generalizes extremely well.

# 2ND TRAINING WITH 2ND LAYER ALSO UNFROZEN

In this stage, the previous best ResNet-50 model (fine-tuned on layers 3–4) was loaded and trained further with deeper fine-tuning, unfreezing layers 2, 3, 4, and the FC head.
This allowed mid-level filters to adapt to textures and contours unique to the office-item dataset while preserving the early low-level features. Other paramets remained the same, with a learnig rate of 5e-6 for the 2nd layer.

    Val Accuracy: 98.8%
    Test Accuracy: 99.3%
    Confusion Matrix: <5 off-class errors per 450 samples per class

Deeper fine-tuning improved low-level adaptability and slightly raised test-set F1 (0.995), confirming strong robustness on unseen data.
However, validation F1 (0.989 vs 0.991) shows that Training 1 remains marginally more stable, while Training 2 generalizes equally well on the held-out test set.
Both models achieved near-perfect recognition, but we decided to keep the "best_resNet50_office_classifier.pth" for future training.

# GUI IMPLEMENTATION FOR TESTING

    ```bash
    cd GUI # go to the directory where the gui.py is found before executing
    python gui.py #to run the app
    ```

After getting "best_resNet50_office_classifier.pth" model, we decided to test it on live camera as well as when uploading images. We built a GUI with customTkinter, where the user can choose either to upload or start camera (camera panel sized to 224x224 px same as what ResNet50 uses). 

classes:
['eraser','glueStick','mouse','paperClip','pen','pencilBox','pencilSharpener','scissor','stapler','waterBottle'] 

Note: It's really important to keep the same order for each classes to avoid misclassifications when testing

Weights are loaded from MODELS_PATH/best_resNet50_office_classifier_2.pth (supports both raw state dict or a checkpoint with model_state_dict).

We used OpenCV to capture frames for live classification, they preprocessed and classified. We used a rolling buffer of 10 to average the prediction, which gives a smoother percentage detection as shown in the live classification when the code is executed.

After testing, a stop camera button is used to stop detection via live camera.

Assets were used to add an aesthetic touch to our window and for better understanding for each button.

Limitations:
    After various tests, we noticed that the model was misclassifying several classes (waterBottle with pencilBox and glueStick, etc..), and sometimes was not able to detect correctly the object and gave fluctuating detection which made it impossible to precisely know if the object was correctly predicted or not.

# RESEARCH ON HOW TO IMPROVE THIS

After analysing our previous results and studying best practices from recent research, we identified that even though our model achieved high accuracy, its robustness and generalization could still be improved. To address this, several key enhancements were introduced:

    1.	Enhanced Data Augmentation
    Stronger training transforms were implemented to increase data diversity and model resilience. These include advanced augmentations that help the model handle variations in lighting, orientation, and occlusions.

    2.	Conditional Class Rebalancing
    Class weights and sampling were adjusted dynamically based on imbalance ratios, ensuring fairer learning across underrepresented classes without overcompensating.

    3.	Selective Layer Fine-Tuning
    The third and fourth layers of the ResNet50 backbone were kept unfrozen. This allows the model to adapt high-level feature extraction to our dataset while retaining stable low-level representations from ImageNet pretraining.

    4.	Resume-Aware Learning Rate (0.3× scaling)
    When resuming from a previously trained checkpoint, the learning rate was scaled down by a factor of 0.3. This smaller step size refines the model’s performance without disrupting previously learned weights.

    5.	Optimizer Upgrade – AdamW
    The optimizer was switched from Adam to AdamW, which decouples weight decay from the learning rate. This modification improves convergence stability and regularization, particularly effective on large datasets.

    6.	Training Tracking and Comparison
    Each epoch’s results (accuracy, loss, F1-score) were stored in .json files for easy comparison, visualization, and reproducibility.

    7.	Duplicate and Leakage Detection
    To ensure data integrity, exact duplicates were identified using MD5 hashing, and near-duplicates were detected using perceptual hashing (aHash) with Hamming distance across the train/validation/test splits.

This prevented data leakage that could artificially inflate model metrics and led to the creation of a cleaner, more reliable dataset.

# RE-TRAINING BEST_RESNET50 WITH NEW PARAMETERS AND MORE ROBUST CODE TO BE ABLE TO GET BETTER RESULTS

After all those modifications and actually running the code, we saw an upgrade in performance.

    F1 improved to become 0.999 thanks to a better regularization (label smoothing, RandomErasing), stability (AdamW + grad clipping), and metric-aligned scheduling (ReduceLROnPlateau on F1).

Resuming the best ResNet50 and upgrading augmentation, loss, optimizer, and LR control pushed performance from ~99.1%/0.991 to 99.89%/0.999 Macro-F1 on test, while the duplicate/leakage audit safeguards metric integrity.

# CHECKING FOR LEAKAGES AND DUPLICATES

After training, we decided to scan all images in train/val/test for split leakages and duplicates. MD5 was used for exact duplicates and aHash for near-duplicates.

Duplicates across splits cause data leakage: the model sees (almost) the same image in training and evaluation, inflating accuracy/F1 and harming true generalization.

From the results: 

1. Exact duplicates

    train & val: 1052
    train & test: 3063
    val & test: 197

2. Near duplicates
 
    train & val: 12 examples
    train & test: 12 examples
    val & test: 25 examples

We audited train/val/test for exact and perceptual duplicates (MD5 + aHash). We found 1,052 (train↔val), 3,063 (train↔test), and 197 (val↔test) exact cross-split duplicates plus many near-duplicates, which can inflate metrics. We cleaned splits to remove leakage and re-trained to report robust performance.

Removing these duplicates ensures that:

1. Evaluation metrics reflect true generalization, not memorization.
2. Model fine-tuning and hyperparameter choices are based on valid feedback.
3. The system performs consistently on unseen, real-world data.

# FIXING LEAKAGES AND DUPLICATES

After cleaning the dataset to remove duplicates and leakage, the model was re-trained using ResNet50 (ImageNet-pretrained) with improved robustness and control mechanisms.

	Data Augmentation: 
    
    Training images were randomly cropped, flipped, color-adjusted, normalized, and randomly erased to improve generalization, while validation and test data used fixed center crops for consistent evaluation.

	Class Imbalance Handling: 
    
    A WeightedRandomSampler was optionally applied when class imbalance exceeded 1.5×.
	
    Fine-Tuning Strategy: 
    
    Early layers remained frozen to preserve learned low-level features, while layer3, layer4, and the fully connected head were trainable to adapt to office-item classification.

	Optimization & Regularization:

	- AdamW optimizer for stability and decoupled weight decay.
	- Label smoothing (0.05) to reduce overconfidence.
	- Learning-rate scheduling (ReduceLROnPlateau) and early stopping to avoid overfitting.
	- Gradient clipping for stable updates.

	Reproducibility: 
    
    All random seeds were fixed (seed=42), and results were saved with consistent naming for full reproducibility.


This script rebuilds the dataset using hash-grouped families so no image (or its byte-identical duplicate) appears in multiple splits, then re-trains a fine-tuned ResNet50 with robust augmentations and modern optimization. It eliminates evaluation leakage, yields trustworthy metrics (accuracy/F1/precision/recall), and saves full artifacts for auditability.

# 3RD AND FINAL TRAINING

In order to remove this leakage and duplicates, we trained our model for a 3rd time.

Some comparisons between our 2nd and 3rd model:

BEFORE: 
    Test: 99.89%

AFTER: 
    Test: 99.80%

This decrease shows that the leakage has been removed, reflecting generalization

What our code does in simple terms:

1. Scans the original `dataset` and computes MD5 for every image to group identical files into families.
2. Each family goes to one and only one split, conflicting families are forced to train only
3. Creates a new dataset `dataset_clean` using symlinks and writes a `splits_clean.json` for tracking.
4. Retraina the same ResNet50 model on the clean splits and exports history, confusion matrices and per-class metrics


Our new model measures performance on truly unseen images, it avoids evaluation contamination(duplicates and near-duplicates) that trains the model to memorize instead of learning generalizable features and it gives stable ablations going forward.

Refer to imageProcessingFinal.ipynb for more detailed documentation.
