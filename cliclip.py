#!/usr/bin/env python3

"""
Find local images based on text prompt.

TODO:
- Add optional dimensionality reduction.
"""

import os
import argparse
import PIL.Image
import torch
from tqdm import tqdm
import clip

# Use Cuda if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', DEVICE)

def main():
    """Main"""
    parser = argparse.ArgumentParser(
        description="Search for images based on textual prompts using CLIP.")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Directory containing the images.")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("-t", "--top", type=int, default=15,
                        help="Number of top scored images to display.")
    parser.add_argument("-s", "--score", action='store_true', help="Display score along output.")
    args = parser.parse_args()

    # Encode text prompt.
    text_encoded = clip.tokenize(args.prompt).to(DEVICE)
    text_features = model.encode_text(text_encoded)

    image_filenames = [f for f in os.listdir(args.directory)
                       if os.path.isfile(os.path.join(args.directory, f))
                       and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
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
    top_scored = min(args.top, len(sorted_images))
    for i, (image_path, score) in enumerate(sorted_images[:top_scored]):
        if args.score:
            print(f"Rank {i+1}: {image_path} (score: {score:.2f})")
        else:
            print(image_path)


if __name__ == '__main__':
    main()
