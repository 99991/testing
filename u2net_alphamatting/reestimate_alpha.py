from PIL import Image
import numpy as np
from scipy.ndimage.morphology import binary_erosion
import requests

image_url = "https://user-images.githubusercontent.com/1842985/94539467-8c03f280-0245-11eb-82d6-a938405b48fe.JPG"

with requests.get(image_url) as r:
    with open("input_image.jpg", "wb") as f:
        f.write(r.content)

size = (1500, 1000)

# Resize image to not run out of memory
Image.open("input_image.jpg").resize(size, Image.BOX).save("input_image.png")

u2net_alpha = Image.open("u2net_prediction.png").convert("L").resize(size, Image.BOX)

# convert to numpy array in range [0, 1]
u2net_alpha = np.array(u2net_alpha)

# guess likely foreground/background
is_foreground = u2net_alpha > 240
is_background = u2net_alpha < 10

# erode foreground/background
size = 31
structure = np.ones((size, size), dtype=np.int)
is_foreground = binary_erosion(is_foreground, structure=structure)
is_background = binary_erosion(is_background, structure=structure, border_value=1)

# build trimap
# 0   = background
# 128 = unknown
# 255 = foreground
trimap = np.full(u2net_alpha.shape, dtype=np.uint8, fill_value=128)
trimap[is_foreground] = 255
trimap[is_background] = 0

Image.fromarray(trimap).save("trimap.png")

from pymatting import cutout

cutout("input_image.png", "trimap.png", "cutout.png")
