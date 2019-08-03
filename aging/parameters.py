#Parameters for neural net training

n_channels = 3
n_encode = 64
n_z = 50
n_l = 10
n_generator = 64
batch_size = 32
image_size = 128
n_discriminator = 16
n_age = int(n_z / n_l)
n_gender = int(n_z / 2)