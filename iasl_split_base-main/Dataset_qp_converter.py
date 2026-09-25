import os
from PIL import Image 
import time

quality = 100
list_of_dirs = ['CAM_FRONT_RIGHT/', 'CAM_FRONT_LEFT/', 'CAM_FRONT/', 'CAM_BACK_RIGHT/', 'CAM_BACK_LEFT/', 'CAM_BACK/',]
lists_of_images = []

for i in range(len(list_of_dirs)):
    lists_of_images.append(os.listdir('og_samples/' + list_of_dirs[i]))

# obj = Image
for i in range(len(lists_of_images)):
    for image in lists_of_images[i]:
        img = Image.open('og_samples/' + list_of_dirs[i] + image)
        img.seek(0)
        img.save('samples/' + list_of_dirs[i] + image, "JPEG", quality=quality)
        # img.close()