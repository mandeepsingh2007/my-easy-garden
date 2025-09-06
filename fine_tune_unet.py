import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --- 1. Configuration ---
DATA_DIR = './data/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PHASE1 = 15  # Epochs for training the decoder only
EPOCHS_PHASE2 = 10  # Epochs for training the entire model
LR_PHASE1 = 1e-3    # Learning rate for the decoder
LR_PHASE2 = 1e-5    # Lower learning rate for the full model
BATCH_SIZE = 8

# --- 2. Data Generation (Creates a simple synthetic dataset) ---
def generate_data(num_samples=100):
    """Generates and saves simple images with circles and their masks."""
    img_dir = os.path.join(DATA_DIR, 'images')
    mask_dir = os.path.join(DATA_DIR, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Generating {num_samples} synthetic samples...")
    for i in tqdm(range(num_samples)):
        # Create a black background
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Draw a random white circle
        center_x = np.random.randint(50, 200)
        center_y = np.random.randint(50, 200)
        radius = np.random.randint(20, 50)
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)
        cv2.circle(mask, (center_x, center_y), radius, 1, -1) # Mask is 0 or 1

        # Add some noise to the image to make it more realistic
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

        cv2.imwrite(os.path.join(img_dir, f'{i}.png'), img)
        cv2.imwrite(os.path.join(mask_dir, f'{i}.png'), mask)

# --- 3. Custom Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Load image (H, W, C) and mask (H, W)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # To RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize image to [0, 1] and add channel dimension to mask
        image = image.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)

        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return torch.tensor(image), torch.tensor(mask)

# --- 4. Training and Validation Logic ---
def train_fn(loader, model, optimizer, loss_fn):
    """A single training epoch."""
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Forward
        predictions = model(images)
        loss = loss_fn(predictions, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def validate_fn(loader, model, loss_fn):
    """A single validation epoch."""
    model.eval()
    total_loss = 0
    dice_score = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            total_loss += loss.item()

            # Calculate Dice Score
            preds_binary = (torch.sigmoid(predictions) > 0.5).float()
            intersection = (preds_binary * masks).sum()
            union = preds_binary.sum() + masks.sum()
            dice_score += (2. * intersection) / (union + 1e-8)

    model.train()
    avg_loss = total_loss / len(loader)
    avg_dice = dice_score / len(loader)
    return avg_loss, avg_dice

# --- 5. Main Execution ---
if __name__ == '__main__':
    # Generate data if it doesn't exist
    if not os.path.exists(DATA_DIR):
        generate_data()

    # Create dataset and dataloaders
    full_dataset = SegmentationDataset(
        image_dir=os.path.join(DATA_DIR, 'images'),
        mask_dir=os.path.join(DATA_DIR, 'masks')
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load pre-trained U-Net with a ResNet34 encoder
    print("Loading pre-trained U-Net model...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1, # Binary segmentation
    ).to(DEVICE)

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss() # Good for binary segmentation

    # --- PHASE 1: Train Decoder Only ---
    print("\n--- Starting Phase 1: Training Decoder ---")
    # Freeze encoder layers
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Optimizer for decoder parameters only
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    
    best_dice_score = -1.0
    for epoch in range(EPOCHS_PHASE1):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, dice_score = validate_fn(val_loader, model, loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE1} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}")
        
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            torch.save(model.state_dict(), 'unet_best_model.pth')
            print("✨ Model saved!")

    # --- PHASE 2: Unfreeze and Train Full Model ---
    print("\n--- Starting Phase 2: Training Full Model ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    # Optimizer for all parameters with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE2)
    
    # Load the best model from Phase 1
    model.load_state_dict(torch.load('unet_best_model.pth'))
    
    for epoch in range(EPOCHS_PHASE2):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, dice_score = validate_fn(val_loader, model, loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE2} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}")
        
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            torch.save(model.state_dict(), 'unet_best_model.pth')
            print("✨ Model saved!")
            
    print(f"\n✅ Training complete! Best Dice Score: {best_dice_score:.4f}")
    print("Best model saved to 'unet_best_model.pth'")