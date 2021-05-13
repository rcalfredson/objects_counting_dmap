import base64
from glob import glob
import json

import numpy as np
from PIL import Image

mapping = {}

all_images = glob('egg_source/heldout/*_dots*')
for label_path in all_images:
    img_path = label_path.replace('_dots.png', '.jpg')
    image = np.array(Image.open(img_path), dtype=np.float32) / 255
    image = np.transpose(image, (2, 0, 1))
    mapping[str(base64.b64encode(np.ascontiguousarray(image.T)))[:100]] = img_path


with open('error_examples/error_key_mapping.json', 'w') as myF:
            json.dump(mapping, myF, ensure_ascii=False, indent=4)