import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import base64
import numpy as np

# Function to convert base64 string to binary
def base64_to_binary(base64_str):
    return base64.b64decode(base64_str)

# Function to convert the model file to base64
def convert_to_base64(file_path):
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read())
        return encoded_string.decode()

# Save the base64 encoded EDSR model to a file (only needed once)
def save_base64_model():
    base64_model = convert_to_base64("EDSR_x4.pb")
    with open("edsr_model_base64.txt", "w") as text_file:
        text_file.write(base64_model)

# Function to browse and load the image
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Function to enhance image quality using super-resolution
def enhance_image():
    image_path = browse_image()
    
    if image_path:
        # Load the image
        image = cv2.imread(image_path)
        
        # Initialize the DNN Super Resolution object
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # Load base64 model string from file
        with open("edsr_model_base64.txt", "r") as file:
            edsr_base64_model = file.read()

        # Convert base64 model to binary and save as .pb file
        model_binary = base64_to_binary(edsr_base64_model)
        with open("EDSR_x4.pb", "wb") as file:
            file.write(model_binary)
        
        # Load the model
        sr.readModel("EDSR_x4.pb")
        sr.setModel("edsr", 4)  # EDSR model with 4x scale factor

        # Apply super resolution to the image
        enhanced_image = sr.upsample(image)
        
        # Save and display the enhanced image
        enhanced_image_path = "enhanced_image.png"
        cv2.imwrite(enhanced_image_path, enhanced_image)

        display_image(enhanced_image_path)

# Function to display the enhanced image in the GUI
def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 400), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    
    image_label.config(image=image_tk)
    image_label.image = image_tk

# Tkinter window
root = tk.Tk()
root.title("Image Enhancement Tool")

# Set the size of the window
root.geometry("500x500")  # Width x Height in pixels

# Set background color
root.configure(bg='lightblue')

# Create the 'Enhance Image' button with color
enhance_button = tk.Button(root, text="Enhance Image", command=enhance_image, bg='darkblue', fg='white', font=('Arial', 12))
enhance_button.pack(pady=20)

# Label to display the image with background color
image_label = tk.Label(root, bg='lightblue')
image_label.pack()

# Run the Tkinter loop
root.mainloop()
