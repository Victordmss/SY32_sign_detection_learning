from supervised_learning.config import *
from supervised_learning.utils import *
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os
import csv
import random
import matplotlib.pyplot as plt
import argparse 

# ------------- IMPORT PARSER & ARGS ---------------

parser = argparse.ArgumentParser()
parser.add_argument('img_folder', metavar='DIR', help='Folder of input images to analyse')
parser.add_argument('pred_folder', metavar='DIR', help='Folder of predicted labels from the analysis')

# Analyser les arguments de la ligne de commande
args = parser.parse_args()


# Load input folder
img_folder = args.img_folder
# Check if input folder exists
if not os.path.exists(img_folder):
    raise FileNotFoundError(f"[Error] Folder '{img_folder}' does not exist.")

# Load input folder
pred_folder = args.pred_folder
# Check if input folder exists
if not os.path.exists(pred_folder):
    raise FileNotFoundError(f"[Error] Folder '{pred_folder}' does not exist.")


def calculate_text_size(text, font):
    # calculate text size based on font properties
    ascent, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = ascent + descent
    return text_width, text_height

def get_brightness(color):
    # Calculate brightness of a color (grayscale value) for the text
    r, g, b = ImageColor.getrgb(color)
    return (r * 299 + g * 587 + b * 114) / 1000 

def visualize_image(filename, csv_filename):
        # Open image
        image_path = filename
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Read bounding box information from CSV file
        if os.path.getsize(csv_filename) > 0:
            with open(csv_filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                #next(csvreader)  # Skip header row
                for row in csvreader:
                    if row:
                        xmin, ymin, xmax, ymax = map(int, row[0:4])
                        class_name = row[4]
        
                        # Define colors for different classes
                        class_colors = {
                            'danger': 'yellow',
                            'interdiction': 'purple',
                            'obligation': 'blue',
                            'stop': 'magenta',
                            'ceder': 'cyan',
                            'frouge': 'red',
                            'forange': 'orange',
                            'fvert': 'green'
                        }
        
                         # Define brightness threshold for determining text color
                        brightness_threshold = 150  
        
                        # Get bounding box color
                        box_color = class_colors.get(class_name, 'white') #white is the de
        
                        # Determine text color based on brightness of box color
                        text_color = 'black' if get_brightness(box_color) > brightness_threshold else 'white'
        
                        # Draw bounding box
                        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=box_color)
        
                        # Define font and size
                        font_size = 30 # Adjust the font size here
                        font = ImageFont.truetype("arial.ttf", font_size)
        
                        # Get text size
                        text_width, text_height = calculate_text_size(class_name, font)
        
                        # Draw filled rectangle as background for class name
                        draw.rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill=box_color)
        
                        # Draw class name text on top of the filled rectangle
                        draw.text((xmin, ymin - text_height), class_name, fill=text_color, font=font)
        return img

# Affiche au hasard des images positives de l'ensemble d'apprentissage
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
for ax in axs.ravel():
    #images_dir = os.path.join("train", "images")
    #labels_dir = os.path.join("train", "labels")
    image_name = random.choice(os.listdir(img_folder))
    image_path = os.path.join(img_folder, image_name)
    csv_path = os.path.join(pred_folder, image_name[:-4] + ".csv")   
    # Call visualize_image function to modify the image
    image_to_display = visualize_image(image_path, csv_path)
    
    # Display the modified image
    ax.imshow(image_to_display)
    ax.axis('off')
        
plt.tight_layout()
plt.show()