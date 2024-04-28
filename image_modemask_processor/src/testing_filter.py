import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


def adjust_brightness(image, factor):
    """ Adjust the brightness of the image.
        Factor > 1.0 makes the image brighter,
        Factor < 1.0 makes the image darker.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """ Adjust the contrast of the image.
        Factor > 1.0 increases contrast,
        Factor < 1.0 decreases contrast.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_gaussian_blur(image, kernel_size):
    """ Apply Gaussian Blur to the image.
        Kernel size can be adjusted to increase/decrease the blur effect.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Load image
image_path = './src/pht1.jpg'
image = Image.open(image_path)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Apply filters
bright_image = adjust_brightness(image, 1.5)  # Increase brightness
dark_image = adjust_brightness(image, 0.5)    # Decrease brightness
contrast_image = adjust_contrast(image, 2.0)  # Increase contrast
blurred_image = apply_gaussian_blur(image_cv, 5)  # Apply blur

# Convert back to PIL Image to save or show
blurred_image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))



# Prepare to display images with matplotlib
fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Create a figure with 4 subplots

# Titles for each subplot
titles = ['Bright Image', 'Dark Image', 'High Contrast Image', 'Blurred Image']

# Images to display
images = [bright_image, dark_image, contrast_image, blurred_image_pil]

for ax, img, title in zip(axs, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')  # Hide axes ticks

plt.show()


# # Display or save the images
# bright_image.show()
# dark_image.show()
# contrast_image.show()
# # blurred_image_pil.show()
