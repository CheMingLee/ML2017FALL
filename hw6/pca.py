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

X = all_imgs.T
X_mean = np.mean(X, axis=0)
U, s, V = np.linalg.svd(X - X_mean, full_matrices=False)

image = io.imread(os.path.join(imgs_path, img))
orgi_shape = np.shape(image)
image = np.array(image.flatten(), dtype=np.float)

X = image.T
X_mean = np.mean(X, axis=0)
k = 4
weight = np.dot(X, U[:,:k])
reconMat = np.dot(U[:,:k], weight) + X_mean

M = reconMat.T.reshape(orgi_shape)
M -= np.min(M)
M /= np.max(M)
M = (M*255).astype(np.uint8)
io.imsave('reconstruction.jpg', M)
