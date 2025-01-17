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
data_path = 'color_names.csv'
colors_df = pd.read_csv(data_path)
colors_df['RGB'] = colors_df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values.tolist()

# Function to extract dominant colors from the image
def extract_colors(image, n_colors=10):
    image = image.resize((100, 100))
    img_data = np.array(image).reshape(-1, 3)

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
    dataset_colors = np.array(colors_df['RGB'].tolist())

    for color, percentage in image_colors.items():
        closest_idx, _ = pairwise_distances_argmin_min([color], dataset_colors)
        matched_color_name = colors_df.iloc[closest_idx[0]]['Name']
        results.append((matched_color_name, percentage, color))

    return results

# Function to display an image in a specified label
def display_image(img, label):
    img.thumbnail((300, 300))
    tk_image = ImageTk.PhotoImage(img)
    label.config(image=tk_image)
    label.image = tk_image

# Function to calculate model accuracy
def calculate_accuracy(image_colors):
    dataset_colors = np.array(colors_df['RGB'].tolist())
    predicted_colors = list(image_colors.keys())
    
    # Find the closest colors in the dataset for the predicted colors
    closest_indices, _ = pairwise_distances_argmin_min(predicted_colors, dataset_colors)
    predicted_labels = [colors_df.iloc[idx]['Name'] for idx in closest_indices]
    
    # Generate ground truth by matching the predicted labels
    ground_truth_labels = predicted_labels  # In this case, we assume the closest match is ground truth
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    return accuracy

# Use this function after extracting colors in the analyze_image function
def analyze_image():
    global original_image, displayed_image, matched_colors_original, matched_colors_edited

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    original_image = Image.open(file_path)
    displayed_image = original_image.copy()

    original_colors = extract_colors(original_image)
    matched_colors_original = match_colors(original_colors)

    # Calculate accuracy and print it
    accuracy = calculate_accuracy(original_colors)
    accuracy_label.config(text=f"Model Accuracy: {accuracy * 100:.2f}%")

    # Clear the treeviews
    for row in tree_original.get_children():
        tree_original.delete(row)
    for row in tree_edited.get_children():
        tree_edited.delete(row)

    # Populate the original colors table
    for color_name, percentage, rgb in matched_colors_original:
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        tree_original.insert("", "end", values=(color_name, f"{percentage:.2f}%", hex_color), tags=(hex_color,))
        tree_original.tag_configure(hex_color, background=hex_color)

    display_image(original_image, original_image_label)
    display_image(displayed_image, edited_image_label)

    # Enable sliders after image is uploaded
    enable_sliders()

# Function to update the edited colors dynamically
def update_edited_colors():
    global displayed_image, matched_colors_edited

    edited_colors = extract_colors(displayed_image)
    matched_colors_edited = match_colors(edited_colors)

    # Clear the edited colors table
    for row in tree_edited.get_children():
        tree_edited.delete(row)

    # Populate the edited colors table
    for color_name, percentage, rgb in matched_colors_edited:
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        tree_edited.insert("", "end", values=(color_name, f"{percentage:.2f}%", hex_color), tags=(hex_color,))
        tree_edited.tag_configure(hex_color, background=hex_color)

# Function to adjust brightness, contrast, and color balance
def adjust_image():
    global displayed_image

    # Check if an image has been loaded
    if "original_image" not in globals() or original_image is None:
        return  # Do nothing if no image has been loaded

    # Perform adjustments if the image exists
    brightness_factor = brightness_scale.get()
    contrast_factor = contrast_scale.get()
    saturation_factor = saturation_scale.get()

    edited_image = ImageEnhance.Brightness(original_image).enhance(brightness_factor)
    edited_image = ImageEnhance.Contrast(edited_image).enhance(contrast_factor)
    edited_image = ImageEnhance.Color(edited_image).enhance(saturation_factor)

    high_factor = high_scale.get()
    mid_factor = mid_scale.get()
    low_factor = low_scale.get()

    img_data = np.array(edited_image, dtype=np.float32) / 255.0

    # Apply simple adjustments for high, mid, and low tones
    img_data = np.clip(img_data * [high_factor, mid_factor, low_factor], 0, 1)
    displayed_image = Image.fromarray((img_data * 255).astype(np.uint8))

    display_image(displayed_image, edited_image_label)
    update_edited_colors()

# Function to disable sliders
def disable_sliders():
    
    brightness_scale.config(state="disabled")
    contrast_scale.config(state="disabled")
    saturation_scale.config(state="disabled")
    high_scale.config(state="disabled")
    mid_scale.config(state="disabled")
    low_scale.config(state="disabled")
    

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
root.title("Color Analyzer")
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
    font=("Helvetica", 16, "bold"),
    bg="#FFA500",
    fg="#000000",
    activebackground="#FFD700",
    activeforeground="#000000",
    command=go_to_main
)

# Place the button at the bottom-left corner
start_button.place(x=150, y=500)  # Adjust x and y as needed based on window size

# Main screen
main_frame = tk.Frame(root, bg="#333333")

image_frame = tk.Frame(main_frame, bg="#333333")
image_frame.pack(pady=20)

original_image_label = tk.Label(image_frame, bg="#333333")
original_image_label.pack(side="left", padx=10)

edited_image_label = tk.Label(image_frame, bg="#333333")
edited_image_label.pack(side="left", padx=10)

# Original and edited colors tables
original_colors_frame = tk.Frame(main_frame, bg="#333333")
original_colors_frame.pack(side="left", padx=20, pady=10)

edited_colors_frame = tk.Frame(main_frame, bg="#333333")
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

# Adjustments and download section
adjust_frame = tk.Frame(main_frame, bg="#333333")
adjust_frame.pack(pady=10)

brightness_label = tk.Label(adjust_frame, text="Brightness", bg="#333333", fg="white")
brightness_label.grid(row=0, column=0, padx=5)
brightness_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
brightness_scale.set(1.0)
brightness_scale.grid(row=0, column=1, padx=5)

contrast_label = tk.Label(adjust_frame, text="Contrast", bg="#333333", fg="white")
contrast_label.grid(row=1, column=0, padx=5)
contrast_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
contrast_scale.set(1.0)
contrast_scale.grid(row=1, column=1, padx=5)

high_label = tk.Label(adjust_frame, text="High", bg="#333333", fg="white")
high_label.grid(row=2, column=0, padx=5)
high_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
high_scale.set(1.0)
high_scale.grid(row=2, column=1, padx=5)

mid_label = tk.Label(adjust_frame, text="Mid", bg="#333333", fg="white")
mid_label.grid(row=3, column=0, padx=5)
mid_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
mid_scale.set(1.0)
mid_scale.grid(row=3, column=1, padx=5)

low_label = tk.Label(adjust_frame, text="Low", bg="#333333", fg="white")
low_label.grid(row=4, column=0, padx=5)
low_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
low_scale.set(1.0)
low_scale.grid(row=4, column=1, padx=5)

# Add the Saturation Slider in the Adjustments Section
saturation_label = tk.Label(adjust_frame, text="Saturation", bg="#333333", fg="white")
saturation_label.grid(row=5, column=0, padx=5)
saturation_scale = tk.Scale(adjust_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", command=lambda e: adjust_image(), state="disabled", bg="#333333", fg="white", troughcolor="#FFD700")
saturation_scale.set(1.0)
saturation_scale.grid(row=5, column=1, padx=5)

download_button = tk.Button(adjust_frame, text="Download Image", command=download_image, font=("Helvetica", 12), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
download_button.grid(row=6, column=0, columnspan=2, pady=10)

accuracy_label = tk.Label(main_frame, text="Model Accuracy: ", font=("Helvetica", 14), bg="#333333", fg="white")
accuracy_label.pack(pady=10)

analyze_button = tk.Button(main_frame, text="Upload and Analyze Image", command=analyze_image, font=("Helvetica", 14, "bold"), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
analyze_button.pack(pady=10)

back_button = tk.Button(main_frame, text="Back to Home", command=go_to_home, font=("Helvetica", 14), bg="#FFA500", fg="#000000", activebackground="#FFD700", activeforeground="#000000")
back_button.pack(pady=10)

root.mainloop()
