import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
import os

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


MIXUP_ALPHA = 0.8
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
INIT_LR = 1e-4

# Auto batch size for MPS/CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 16 if device.type in ['mps', 'cpu'] else 32
print(f"🧠 Using device: {device} | Batch size: {BATCH_SIZE}")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dir = '/Users/adrianpothanah/Coursework1-Robotics/RoboticsC1-PDE-3802/dataset/train'
val_dir   = '/Users/adrianpothanah/Coursework1-Robotics/RoboticsC1-PDE-3802/dataset/val'
test_dir  = '/Users/adrianpothanah/Coursework1-Robotics/RoboticsC1-PDE-3802/dataset/test'

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data   = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_data  = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

num_classes = len(train_data.classes)
print(f"📚 Classes detected: {train_data.classes}")


model = models.resnet50(pretrained=True)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few blocks
for name, child in model.named_children():
    if name in ['layer4', 'avgpool', 'fc']:
        for param in child.parameters():
            param.requires_grad = True

# Add dropout + new classifier
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-4},
])

# OneCycleLR for smooth learning rate adaptation
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


best_f1 = 0.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    # Progressive unfreezing
    if epoch == 5:
        for param in model.layer3.parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': model.layer3.parameters(), 'lr': 5e-5})
        print("🟢 Unfroze layer3 at epoch 5")

    if epoch == 10:
        for param in model.layer2.parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': model.layer2.parameters(), 'lr': 1e-5})
        print("🟢 Unfroze layer2 at epoch 10")

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        # MixUp augmentation
        inputs, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() +
                    (1 - lam) * predicted.eq(targets_b).sum().item())

        progress.set_postfix({
            "Loss": f"{running_loss / (total / BATCH_SIZE):.4f}",
            "Acc": f"{100. * correct / total:.2f}%"
        })

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/len(train_loader):.4f}")


    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Validation F1: {val_f1:.3f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_resNet50_final_mixup_onecycle.pth')
        print(f"✅ Best model saved (F1 = {best_f1:.3f})")
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print("⏹ Early stopping triggered.")
            break

print("\nEvaluating on test set...")
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds, average='macro')
conf_matrix = confusion_matrix(test_labels, test_preds)
print(f"🎯 Test Macro F1: {test_f1:.3f}")
print("Confusion Matrix:\n", conf_matrix)

torch.save(model.state_dict(), 'resNet50_final_trained_model.pth')
print("💾 Final model saved as 'resNet50_final_trained_model.pth'")