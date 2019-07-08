import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())
fig = plt.figure(figsize=(8, 8))
for index, image_path in enumerate(np.random.choice(os.listdir('data/UTKFace/'), 9)):
    image = plt.imread(os.path.join('data/UTKFace', image_path))
    fig.add_subplot(3, 3, index+1)
    plt.imshow(image)
plt.show()

