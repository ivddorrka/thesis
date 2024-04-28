import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image as PILImage
from model import UNET
import os 
import numpy as np

DEVICE = "cpu"
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
LEARNING_RATE = 1e-5

import torch

def load_model(model, model_path, map_location='cpu'):
    # Load the checkpoint
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=map_location)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model



def predict_and_save_image(model, image, destination_folder="./", device="cuda"):
    # Load the image
    original_size = image.size
    print(f"Original size is {original_size}")
    device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_HEIGHT)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43364812, 0.46881031, 0.42198243], std=[0.22990168, 0.23807067, 0.23866756]),
    ])

    input_data = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        preds = model(input_data)

    _, pred_class = preds.max(dim=1)

    class_colors = {
            0: [0, 0, 0],    # Non-labeled (black)
            1: [165, 42, 42],  # Class A (red)
            2: [0, 100, 0]  # Class B (blue)
        }
            
    pred_color_img = torch.zeros((pred_class.shape[0], 3, *pred_class.shape[1:]), dtype=torch.float32, device=device)
    for class_idx, color in class_colors.items():
        
        mask = (pred_class == class_idx).unsqueeze(1).float().to(device=device)
        color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 3, 1, 1).to(device=device)
        pred_color_img += mask * color_tensor

    # torchvision.utils.save_image(
    #     pred_color_img / 255.0, destination_folder
    # )

    pred_color_np = pred_color_img[0].cpu().numpy().transpose(1, 2, 0)

    # Ensure the array has the correct shape (remove singleton dimensions)
    pred_color_np = np.squeeze(pred_color_np)

    pred_color_np = (pred_color_np).astype(np.uint8)
    pred_color_pil = PILImage.fromarray(pred_color_np).resize(original_size)

    
    return pred_color_pil


# def compare_images(img1, img2):
#     # Ensure both images are the same size
#     if img1.size != img2.size:
#         print("Images are not the same size.")
#         return
    
#     # Convert images to the same mode if necessary, e.g., both to RGB
#     if img1.mode != img2.mode:
#         img1 = img1.convert('RGB')
#         img2 = img2.convert('RGB')

#     pixels1 = img1.load()
#     pixels2 = img2.load()
    
#     matching_pixels = 0
#     total_pixels = img1.size[0] * img1.size[1]

#     # Iterate over each pixel and compare
#     for x in range(img1.size[0]):
#         for y in range(img1.size[1]):
#             if pixels1[x, y] == pixels2[x, y]:
#                 matching_pixels += 1

#     # Calculate percentage of matching pixels
#     matching_percentage = (matching_pixels / total_pixels) * 100
    
#     # print(f"Percentage of matching pixels: {matching_percentage:.2f}%")
#     return matching_percentage



# # # Example usage
# # model = UNET(in_channels=3, out_channels=3).to(DEVICE)
# # loaded_model = load_model(model, './src/trained_model_epoch675.pth', map_location='cpu')
# # # model.load_state_dict(torch.load('./trained_model_epoch150.pth', map_location=torch.device('cpu')))
# # # model.load_state_dict(torch.load('trained_model_epoch150.pth'))
# # loaded_model.to("cpu")  # Assuming CUDA is available

# # # for img_pth in 
# # dir1 = "/home/mrs/coco2unet/png_zp/png_originals"
# # dir2 = "/home/mrs/coco2unet/png_zp/masks_png"


# # files_dir1 = os.listdir(dir1)
# # total_matches = 0
# # num_matches = 0
# # for fnmae in files_dir1:
# #     fnmae_2 = os.path.join(dir2, fnmae)
# #     gname_path = os.path.join(dir1, fnmae)
# #     if os.path.exists(fnmae_2):

# #         image = PILImage.open(gname_path).convert("RGB")   
# #         prediction_mask = predict_and_save_image(loaded_model, image, destination_folder=f'predicted_masks/{fnmae}')
# #         img2 = PILImage.open(fnmae_2)
            
# #         # Compare the images
# #         if prediction_mask.size != img2.size:
# #             print(f"{gname_path}: Images are not the same size.")
# #             continue
        
# #         # Ensure both images are in the same mode
# #         if prediction_mask.mode != img2.mode:
# #             prediction_mask = prediction_mask.convert('RGB')
# #             img2 = img2.convert('RGB')

# #         pixels1 = prediction_mask.load()
# #         pixels2 = img2.load()

# #         matching_pixels = 0
# #         total_pixels = prediction_mask.size[0] * prediction_mask.size[1]

# #         # Iterate over each pixel and compare
# #         for x in range(prediction_mask.size[0]):
# #             for y in range(prediction_mask.size[1]):
# #                 if pixels1[x, y] == pixels2[x, y]:
# #                     matching_pixels += 1

# #         # img2.save(f'predicted_masks/{fnmae}')


# #         # Calculate percentage of matching pixels
# #         matching_percentage = (matching_pixels / total_pixels) * 100
# #         print(f"{fnmae}: Percentage of matching pixels: {matching_percentage:.2f}%")
# #         total_matches += matching_percentage
# #         num_matches += 1

# # result_now = total_matches / num_matches
# # print(result_now)



# # image_path = './src/pht4.jpg'
#     # image = PILImage.open(image_path).convert("RGB")   
#     # predict_and_save_image(loaded_model, image)
