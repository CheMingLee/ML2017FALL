import numpy as np
from skimage import io
import sys
import os

imgs_path = sys.argv[1]
img = sys.argv[2]

all_imgs = []
for i in range(415):
	image = io.imread(os.path.join(imgs_path, '{0}.jpg'.format(i)))
	all_imgs.append(image.flatten())

all_imgs = np.array(all_imgs, dtype=np.float)

X_mean = np.mean(all_imgs, axis=0)
X_norm = all_imgs - X_mean
U, s, V = np.linalg.svd(X_norm.T, full_matrices=False)

image = io.imread(os.path.join(imgs_path, img))
orgi_shape = np.shape(image)
X = np.array(image.flatten(), dtype=np.float)

k = 4
weight = np.dot(X - X_mean, U[:,:k])
reconMat = np.dot(U[:,:k], weight) + X_mean

M = reconMat.T.reshape(orgi_shape)
M -= np.min(M)
M /= np.max(M)
M = (M*255).astype(np.uint8)
io.imsave('reconstruction.jpg', M)
