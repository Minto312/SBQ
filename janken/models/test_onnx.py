import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import onnxruntime
from PIL import Image
import numpy as np
session = onnxruntime.InferenceSession("model.onnx")
image = Image.open("test.jpg")
image = np.array(image, dtype=np.float32)
image = image.transpose(2, 0, 1)
image = np.expand_dims(image, axis=0)
image[0,...] = 1
output = session.run(["output"], {"input":image })

print(output)