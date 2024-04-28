import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import os 
from torchvision.utils import save_image
import torch.nn.functional as F
from utils import load_checkpoint, save_checkpoint,   get_loaders, check_accuracy, save_predictions_as_imgs  



# Hyper parameters

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = './png_zp/png_originals'
TRAIN_MASK_DIR = './png_zp/masks_png'
VAL_IMG_DIR = './png_zp/val_ors_png'
VAL_MASK_DIR = './png_zp/masks_validation'
LOSS_RESULTS = []

def targets_to_indices(targets):
    # Map target labels (0, 1, 100) to class indices (0, 1, 2)
    class_indices = torch.zeros_like(targets, dtype=torch.long)
    class_indices[targets == 1] = 1
    class_indices[targets == 2] = 2
    
    return class_indices


def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader)
    loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE, dtype=torch.long)  

        
        with torch.cuda.amp.autocast():
            predictions =  model(data)

            targets = targets_to_indices(targets)
            loss = nn.CrossEntropyLoss()(predictions, targets)
        
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())
        
    LOSS_RESULTS.append(loss.item())
        
        
#current:
#Overall mean: [0.43364812 0.46881031 0.42198243]
# Overall std: [0.22990168 0.23807067 0.23866756]

def evaluate_fn(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=DEVICE, dtype=torch.long)  

            predictions =  model(data)
            targets = targets_to_indices(targets)
            
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

    model.train()
    return val_loss / len(loader)


def save_final_model(model, optimizer, save_path):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"Current trained model saved at {save_path}")


import matplotlib.pyplot as plt

def save_plot_loss_function(num_epochs, v_l, t_l):
    print(f"len(v_l) = {len(v_l)}, len(t_l) = {len(t_l)}")
    x_values = range(max(len(v_l), len(t_l)))
    print(f"len(x_values) = {len(x_values)}")
    plt.plot(x_values, v_l, label='Validation loss')
    plt.plot(x_values, t_l, label='Training loss')

    plt.xlabel('Epochs')
    plt.ylabel('loss results')
    plt.legend()

    save_path = 'saved_images/lf_plot.png'
    plt.savefig('saved_images/lf_plot.png')
    plt.clf()
    print(f"Plot saved to {save_path}")
    
    
    
def main():
    print(BATCH_SIZE)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.43364812, 0.46881031, 0.42198243],
                std=[0.22990168, 0.23807067, 0.23866756],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.43364812, 0.46881031, 0.42198243],
                std=[0.22990168, 0.23807067, 0.23866756],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    
    
    
    
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)

    checkpoint_path = 'saved_models/trained_model_epoch550.pth'

    checkpoint_path = "my_checkpoint.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust parameters as needed
    optimizer.load_state_dict(checkpoint['optimizer'])

    

    loss_fn = nn.CrossEntropyLoss()
    
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY)
    
    scaler = torch.cuda.amp.GradScaler()
    v_l = []
    print("changed code")
    for epoch in range(551, NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
    
    
        with torch.no_grad():
            val_loss = evaluate_fn(val_loader, model, loss_fn, device=DEVICE)
            v_l.append(val_loss)
            
        print(f"Epoch {epoch}, Training Loss: {LOSS_RESULTS[-1]}, Validation Loss: {val_loss}")
        
        save_plot_loss_function(epoch, v_l, LOSS_RESULTS)
        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        
#         check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        if (epoch%5 == 0 and epoch!=0) or epoch==NUM_EPOCHS-1:  # Save the model after the last epoch
            save_path = f"saved_models/trained_model_epoch{epoch}.pth"
            print(f"Saving current state of the model, path = {save_path}")
            save_final_model(model, optimizer, save_path)
        
        if epoch%50==0 and epoch !=0:
            
            print("TRAIN LOSS")
            print(LOSS_RESULTS)
            print()
            print("VAL LOSS")
            print(v_l)
        
    
    
# if __name__=="__main__":
#     main()
