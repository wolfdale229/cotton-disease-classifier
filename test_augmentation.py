'''Testing some augmentation steps of the training data'''

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

abspath = os.path.abspath('.')
image_path = os.path.join(abspath, 'train')
image_file = os.path.join(image_path, 'diseased cotton leaf/dis_leaf (1)_iaip.jpg')

img = image.load_img(image_file, target_size=(224, 224), interpolation='nearest')
img = image.img_to_array(img, dtype=np.float32)
img /=225

plt.imshow(img)
plt.show()
      