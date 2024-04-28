import torch
import torchvision
from dataset import ManaualDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
    

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_ds = ManaualDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    
    val_ds = ManaualDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )
    val_loader= DataLoader(
       val_ds,
       batch_size=batch_size,
       num_workers = num_workers,
       pin_memory=pin_memory,
       shuffle=False
    )
    
    return train_loader, val_loader


    
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
        
        _, pred_class = preds.max(dim=1)
        
        class_colors = {
            0: [0, 0, 0],    
            1: [165, 42, 42], 
            2: [0, 100, 0]  
        }
            
        pred_color_img = torch.zeros((pred_class.shape[0], 3, *pred_class.shape[1:]), dtype=torch.float32, device=device)
        for class_idx, color in class_colors.items():
            # print(f"Class idx = {class_idx}, color = {color}")
            # print(f"Pre class unqiue = {torch.unique(pred_class)}")
            
            mask = (pred_class == class_idx).unsqueeze(1).float().to(device=device)
            color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 3, 1, 1).to(device=device)
            pred_color_img += mask * color_tensor

        torchvision.utils.save_image(
            pred_color_img / 255.0, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")


    model.train()
    