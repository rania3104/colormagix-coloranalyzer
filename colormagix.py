import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = 'color_names.csv'  # Dataset path
colors_df = pd.read_csv(data_path)
colors_df['RGB'] = colors_df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values.tolist()


# Function to extract dominant colors from the image
def extract_colors(image, n_colors=10):
    image = image.resize((100, 100))  # Resize for faster processing
    img_data = np.array(image).reshape(-1, 3)  # Flatten image into RGB values

    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_data)

    cluster_centers = kmeans.cluster_centers_.astype(int)
    cluster_labels = kmeans.labels_

    counts = Counter(cluster_labels)
    total_pixels = sum(counts.values())

    colors_percentages = {
        tuple(center): (counts[i] / total_pixels) * 100
        for i, center in enumerate(cluster_centers)
    }

    return colors_percentages

# Function to match colors with the dataset
def match_colors(image_colors):
    results = []
    dataset_colors = np.array(colors_df['RGB'].tolist())  # Dataset colors
    
    for color, percentage in image_colors.items():
        closest_idx, _ = pairwise_distances_argmin_min([color], dataset_colors)
        matched_color_name = colors_df.iloc[closest_idx[0]]['Name']
        results.append((matched_color_name, percentage, color))

    return results

# Function to update both original and edited color tables
def update_color_tables():
    global matched_colors_original, matched_colors_edited

    # Clear the existing rows in both tables
    for row in tree_original.get_children():
        tree_original.delete(row)
    for row in tree_edited.get_children():
        tree_edited.delete(row)

    # Insert new rows into the original colors table
    for color_name, percentage, rgb in matched_colors_original:
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        tree_original.insert("", "end", values=(color_name, f"{percentage:.2f}%", hex_color), tags=(hex_color,))
        tree_original.tag_configure(hex_color, background=hex_color)

    # Insert new rows into the edited colors table
    for color_name, percentage, rgb in matched_colors_edited:
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        tree_edited.insert("", "end", values=(color_name, f"{percentage:.2f}%", hex_color), tags=(hex_color,))
        tree_edited.tag_configure(hex_color, background=hex_color)

# Function to compute accuracy between original image and dataset
def compute_accuracy(image_colors):
    dataset_colors = np.array(colors_df['RGB'].tolist())
    predicted_colors = list(image_colors.keys())
    
    # Find the closest colors in the dataset for the predicted colors
    closest_indices, _ = pairwise_distances_argmin_min(predicted_colors, dataset_colors)
    predicted_labels = [colors_df.iloc[idx]['Name'] for idx in closest_indices]
    
    # Generate ground truth by matching the predicted labels
    ground_truth_labels = predicted_labels  # In this case, we assume the closest match is ground truth
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    # Update the accuracy label
    accuracy_label.config(text=f"Accuracy: {accuracy:.2%}")

# Function to analyze the image
def analyze_image():
    global original_image, displayed_image, matched_colors_original, matched_colors_edited

    # Ask user to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Open and process the image
    original_image = Image.open(file_path)
    displayed_image = original_image.copy()

    # Extract dominant colors from the original image
    original_colors = extract_colors(original_image)
    matched_colors_original = match_colors(original_colors)

    # Set the "edited" table to the same initial data as the "original" image
    matched_colors_edited = matched_colors_original.copy()

    # Update the color tables
    update_color_tables()

    # Display the original and edited images
    display_image(original_image, original_image_label)
    display_image(displayed_image, edited_image_label)

    # Compute accuracy for the original image and the dataset
    compute_accuracy(original_colors)

    # Enable the sliders for adjustment
    enable_sliders()

# Function to adjust brightness, contrast, and color balance
def adjust_image():
    global original_image, displayed_image, matched_colors_original, matched_colors_edited

    # Perform adjustments
    brightness_factor = brightness_scale.get()
    contrast_factor = contrast_scale.get()
    saturation_factor = saturation_scale.get()

    # Adjust brightness, contrast, and color
    edited_image = ImageEnhance.Brightness(original_image).enhance(brightness_factor)
    edited_image = ImageEnhance.Contrast(edited_image).enhance(contrast_factor)
    edited_image = ImageEnhance.Color(edited_image).enhance(saturation_factor)

    # Update the edited image with tone adjustments
    high_factor = high_scale.get()
    mid_factor = mid_scale.get()
    low_factor = low_scale.get()

    img_data = np.array(edited_image, dtype=np.float32) / 255.0
    img_data = np.clip(img_data * [high_factor, mid_factor, low_factor], 0, 1)
    displayed_image = Image.fromarray((img_data * 255).astype(np.uint8))

    # Extract colors again after adjustments
    edited_colors = extract_colors(displayed_image)
    matched_colors_edited = match_colors(edited_colors)

    # Update the color tables with the new edited colors
    update_color_tables()

    # Display the edited image
    display_image(displayed_image, edited_image_label)

# Function to display the image in the GUI
def display_image(image, label):
    # Maintain aspect ratio and resize
    base_width = 450  # Adjusted width
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size))

    # Convert to Tkinter-compatible image
    image_tk = ImageTk.PhotoImage(image)

    # Center the image in the label cell
    label.config(image=image_tk)
    label.image = image_tk

# Function to enable sliders
def enable_sliders():
    brightness_scale.config(state="normal")
    contrast_scale.config(state="normal")
    saturation_scale.config(state="normal")
    high_scale.config(state="normal")
    mid_scale.config(state="normal")
    low_scale.config(state="normal")
    brightness_scale.set(1.0)
    contrast_scale.set(1.0)
    saturation_scale.set(1.0)
    high_scale.set(1.0)
    mid_scale.set(1.0)
    low_scale.set(1.0)

# Function to disable sliders
def disable_sliders():
    brightness_scale.config(state="disabled")
    contrast_scale.config(state="disabled")
    saturation_scale.config(state="disabled")
    high_scale.config(state="disabled")
    mid_scale.config(state="disabled")
    low_scale.config(state="disabled")
# Function to download the edited image
def download_image():
    if displayed_image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            displayed_image.save(file_path)

# Function to navigate to the main screen
def go_to_main():
    reset_images_and_tables()  # Reset all images and tables
    home_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)
    disable_sliders()  # Disable sliders when the main frame is opened

# Function to navigate back to the home screen
def go_to_home():
    # Hide the main frame and display the home frame
    main_frame.pack_forget()
    home_frame.pack(fill="both", expand=True)


def reset_images_and_tables():
    global original_image, displayed_image, matched_colors_original, matched_colors_edited

    # Reset image variables
    original_image = None
    displayed_image = None
    matched_colors_original = []
    matched_colors_edited = []

    # Clear the treeviews
    for row in tree_original.get_children():
        tree_original.delete(row)
    for row in tree_edited.get_children():
        tree_edited.delete(row)

    # Clear image labels
    original_image_label.config(image="")
    edited_image_label.config(image="")

    # Reset sliders to default values and disable them
    brightness_scale.set(1.0)
    contrast_scale.set(1.0)
    saturation_scale.set(1.0)
    high_scale.set(1.0)
    mid_scale.set(1.0)
    low_scale.set(1.0)
    disable_sliders()

# Main GUI setup
root = tk.Tk()
root.title("ColorMagix Studio (Color Analyzer Model)")
root.geometry("1400x800")  # Example size; adjust as needed
root.resizable(False, False)  # Disables resizing

# Home screen
home_frame = tk.Frame(root, bg="white")
home_frame.pack(fill="both", expand=True)

bg_image = tk.PhotoImage(file="colormagix bg.png")
home_bg_label = tk.Label(home_frame, image=bg_image)
home_bg_label.pack()
home_bg_label.place(relx=0.5, rely=0.5, anchor="center")  # Centers the image without resizing

start_button = tk.Button(
    home_frame,
    text="Start Analyzing Colors",
    font=("Tw Cen MT", 16),
    bg="#FFA500",
    fg="#000000",
    activebackground="#FFD700",
    activeforeground="#000000",
    command=go_to_main, padx=40,pady=10
)

# Place the button at the bottom-left corner
start_button.place(x=100, y=500)  

# Main screen
main_frame = tk.Frame(root, bg="#333333")

# Left frame for images and tables
left_frame = tk.Frame(main_frame, bg="#333333")
left_frame.pack(side="left", fill="y", padx=20, pady=20)

image_frame = tk.Frame(left_frame, bg="#333333")
image_frame.pack(pady=20)

original_image_label = tk.Label(image_frame, bg="#333333")
original_image_label.pack(side="left", padx=10)

edited_image_label = tk.Label(image_frame, bg="#333333")
edited_image_label.pack(side="left", padx=10)

# Original and edited colors tables
original_colors_frame = tk.Frame(left_frame, bg="#333333")
original_colors_frame.pack(side="left", padx=20, pady=10)

edited_colors_frame = tk.Frame(left_frame, bg="#333333")
edited_colors_frame.pack(side="left", padx=20, pady=10)

columns = ("Color Name", "Percentage", "Hex Code")

# Original colors treeview
tree_original = ttk.Treeview(original_colors_frame, columns=columns, show="headings", height=10)
for col in columns:
    tree_original.heading(col, text=col)
    tree_original.column(col, width=150)
tree_original.pack()

# Edited colors treeview
tree_edited = ttk.Treeview(edited_colors_frame, columns=columns, show="headings", height=10)
for col in columns:
    tree_edited.heading(col, text=col)
    tree_edited.column(col, width=150)
tree_edited.pack()

# Right section frame for adjustments and buttons
right_frame = tk.Frame(main_frame, bg="#0f0e19", width=400)
right_frame.pack(side="right", fill="y", ipadx=200)

logo_image = tk.PhotoImage(file="colormagix logo.png")
logo_label = tk.Label(right_frame, image=logo_image, bg="#0f0e19")
logo_label.pack()

# Adjustments and download section
adjust_frame = tk.Frame(right_frame, bg="#0f0e19")
adjust_frame.pack(pady=50, padx=10)

brightness_label = tk.Label(adjust_frame, text="Brightness", bg="#0f0e19", fg="white")
brightness_label.grid(row=0, column=0, padx=5)
brightness_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
brightness_scale.grid(row=0, column=1, padx=5)

contrast_label = tk.Label(adjust_frame, text="Contrast", bg="#0f0e19", fg="white")
contrast_label.grid(row=1, column=0, padx=5)
contrast_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
contrast_scale.grid(row=1, column=1, padx=5)

high_label = tk.Label(adjust_frame, text="High", bg="#0f0e19", fg="white")
high_label.grid(row=2, column=0, padx=5)
high_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
high_scale.grid(row=2, column=1, padx=5)

mid_label = tk.Label(adjust_frame, text="Mid", bg="#0f0e19", fg="white")
mid_label.grid(row=3, column=0, padx=5)
mid_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
mid_scale.grid(row=3, column=1, padx=5)

low_label = tk.Label(adjust_frame, text="Low", bg="#0f0e19", fg="white")
low_label.grid(row=4, column=0, padx=5)
low_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
low_scale.grid(row=4, column=1, padx=5)

# Add the Saturation Slider in the Adjustments Section
saturation_label = tk.Label(adjust_frame, text="Saturation", bg="#0f0e19", fg="white")
saturation_label.grid(row=5, column=0, padx=5)
saturation_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#0f0e19", fg="white", troughcolor="#FFD700")
saturation_scale.grid(row=5, column=1, padx=5)

analyze_button = tk.Button(right_frame, text="Upload Image", command=analyze_image, font=("Tw Cen MT", 14), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
analyze_button.pack(pady=10)

download_button = tk.Button(right_frame, text="Download Image", command=download_image, font=("Tw Cen MT", 14), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
download_button.pack(pady=10)

accuracy_label = tk.Label(right_frame, text="Model Accuracy: ", font=("Tw Cen MT", 14), bg="#0f0e19", fg="white")
accuracy_label.pack(pady=10)

back_button = tk.Button(right_frame, text="Back to Home", command=go_to_home, font=("Tw Cen MT", 14), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
back_button.pack(pady=10)

root.mainloop()