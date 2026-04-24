from PIL import Image, ImageEnhance
import sys

# input_path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
# output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
input_path = r"C:\Users\samee\Downloads\WhatsApp Image 2026-04-24 at 11.30.34 AM.jpeg"
output_path = r"C:\Users\samee\Downloads\processed_image.jpg"

img = Image.open(input_path).convert('L')
 
# Enhance contrast
img = ImageEnhance.Contrast(img).enhance(3.0)
 
# Whiten background: push pixels above threshold to pure white
import numpy as np
arr = np.array(img)
arr[arr > 120] = 255
img = Image.fromarray(arr)
 
# Resize to ~400x200 preserving aspect ratio
img.thumbnail((400, 200), Image.LANCZOS)
 
img.save(output_path)
print(f"Saved to {output_path} — size: {img.size}")