import os
import argparse
import PIL.Image
import torch
from tqdm import tqdm
import clip

parser = argparse.ArgumentParser(description="Search for images based on textual prompts using CLIP.")
parser.add_argument("--directory", type=str, required=True, help="Directory containing the images.")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
parser.add_argument("--top", type=int, default=15, help="Number of top scored images to display.")
args = parser.parse_args()

# Use Cuda if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', DEVICE)

# Encode text prompt.
text_encoded = clip.tokenize(args.prompt).to(DEVICE)
text_features = model.encode_text(text_encoded)

image_filenames = [f for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
image_scores = []

for image_filename in tqdm(image_filenames, desc="Processing images"):

    # Load image.
    image_path = os.path.join(args.directory, image_filename)
    image = PIL.Image.open(image_path)

    # Preprocess image.
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Encode image.
    image_features = model.encode_image(image_tensor)

    # Calculate the similarity score between the image and the text prompt.
    score = (100.0 * image_features @ text_features.T).squeeze(0)
    image_scores.append((image_path, score.item()))

# Sort the images based on their similarity scores.
sorted_images = sorted(image_scores, key=lambda x: x[1], reverse=True)

# Display the top scored images along with their full paths.
n = min(args.top, len(sorted_images))
for i, (image_path, score) in enumerate(sorted_images[:n]):
    print(f"Rank {i+1}: {image_path} (score: {score:.2f})")
