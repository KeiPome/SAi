import cv2
import os
from tqdm import tqdm

def extract_frames(video_path: str, save_dir: str = None):
    """
    この関数は、指定されたビデオファイル(`video_path`)からフレームを抽出し、
    画像ファイルとして保存します。処理の進捗はtqdmプログレスバーで表示されます。

    引数:
        video_path (str): 処理対象のビデオファイルへのパス。
        save_dir (str, optional): 抽出した画像を保存するディレクトリ。指定しない場合(`None`)は、ビデオファイルのあるディレクトリの親ディレクトリの下に "img/{ビデオファイル名}" というディレクトリが作成され、そこに保存されます。

    戻り値:
        str: 画像が保存されたディレクトリのパス。

    """
    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Default save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(video_path), "../img", video_name)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file: {video_path}")

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # Display progress bar
    with tqdm(total=total_frames, desc="Extracting Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create filename in format img000000001.jpg
            img_filename = f"img{frame_count:09d}.jpg"
            img_path = os.path.join(save_dir, img_filename)

            # Save the frame as an image
            cv2.imwrite(img_path, frame)
            frame_count += 1
            pbar.update(1)  # Update progress bar

    cap.release()
    print(f"\n✅ Extraction complete: {frame_count} frames saved to {save_dir}")

    return save_dir


import os
import json
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

def load_json(json_path):
    """Load existing JSON file or return an empty dictionary if not found."""
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            return json.load(file)
    return {}

def save_json(json_path, data):
    """Save data to JSON file."""
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

def select_images(image_dir):
    """
    List images in the directory and allow the user to select ones for labeling.

    Parameters:
        image_dir (str): Directory containing images.

    Returns:
        list: List of selected image paths.
    """
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("No images found in the directory.")
        return []

    print("\nAvailable images:")
    for i, img in enumerate(images):
        print(f"{i + 1}. {img}")

    selected_indices = input("\nEnter the numbers of the images you want to label (comma-separated): ")
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
        selected_images = [os.path.join(image_dir, images[i]) for i in selected_indices if 0 <= i < len(images)]
        return selected_images
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        return []

def choose_label(labels):
    """
    Display a simple dialog box for label selection.

    Parameters:
        labels (list): List of possible labels.

    Returns:
        str: The selected label.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    label = simpledialog.askstring("Label Selection", f"Enter a label ({', '.join(labels)}):")
    return label if label in labels else None

def label_image(image_path, json_path, labels):
    """
    Open an image in GUI, allow the user to click points and select a predefined label.
    Saves the labeled points to a JSON file.

    Parameters:
        image_path (str): Path to the image to be labeled.
        json_path (str): Path to the JSON file where labels will be stored.
        labels (list): Predefined list of labels to choose from.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

    data = load_json(json_path)

    if image_path not in data:
        data[image_path] = []

    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            label = choose_label(labels)  # Use dialog to select a label
            if label:
                points.append({"x": x, "y": y, "label": label})
                plt.scatter(x, y, color='red', s=30)
                plt.text(x, y, label, color='red', fontsize=12, verticalalignment='bottom')
                plt.draw()

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Click to label points on: {os.path.basename(image_path)}")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if points:
        data[image_path].extend(points)
        save_json(json_path, data)
        print(f"✅ Labels saved to {json_path}")

def label_selected_images(image_dir, json_path, labels):
    """
    Allow user to select specific images from a directory for labeling.

    Parameters:
        image_dir (str): Directory containing images.
        json_path (str): Path to JSON file for saving labeled data.
        labels (list): List of predefined labels.
    """
    selected_images = select_images(image_dir)

    if not selected_images:
        print("No images selected. Exiting.")
        return

    for image in selected_images:
        label_image(image, json_path, labels)
