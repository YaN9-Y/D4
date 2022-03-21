import numpy as np
from PIL import Image
import argparse
import os

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str )
parser.add_argument('--output_path', type=str)
parser.add_argument('--sigma', type=int)

args= parser.parse_args()

input_path = args.input_path
output_path = args.output_path
sigma = args.sigma

original_images = os.listdir(input_path)

for item in original_images:
    if item[-3:] in ['jpg', 'png']:
        img_clean = np.array(Image.open(os.path.join(input_path, item)))
        normal_noise = np.random.normal(0, sigma, size=(img_clean.shape[0],img_clean.shape[1], img_clean.shape[2]))
        img_noisy = img_clean + normal_noise

        img_noisy = np.clip(img_noisy, a_min=0, a_max=255).astype(np.uint8)

        img_noisy = Image.fromarray(img_noisy)

        img_noisy.save(os.path.join(output_path, item))

        print(item + ' complete')




