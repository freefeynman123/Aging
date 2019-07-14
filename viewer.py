import matplotlib.pyplot as plt
import numpy as np
import os

DESTINATION_DIR = 'data/renamed'

fig = plt.figure(figsize=(8, 8))
for index, image_path in enumerate(np.random.choice(os.listdir(DESTINATION_DIR), 9)):
    image = plt.imread(os.path.join(DESTINATION_DIR, image_path))
    print(image.shape)
    fig.add_subplot(3, 3, index+1)
    plt.imshow(image)
plt.show()

