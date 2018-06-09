import numpy as np # Array manipulation
from robust_pca import robust_pca


def corrupt(image, ratio):
    h, w, c = image.shape
    to_destroy = int(h * w * ratio)
    perturbed = np.copy(image)
    for i in range(c):
        layer = image[:,:,i]
        indices = np.random.choice(np.arange(h * w), to_destroy, replace=False)
        values = np.random.randint(0, 255 + 1, to_destroy)
        result = np.ravel(layer)
        result[indices] = values
        perturbed[:,:,i] = np.reshape(result, (h, w))

    return perturbed


def recover(image, max_iters=100):
    return np.stack([robust_pca(image[:,:,i], max_iters)[0]
                     for i in range(image.shape[2])], axis=-1).astype(np.uint8)


def similarity(im1, im2):
    centeredA = im1 - np.mean(im1, axis=(0,1))
    centeredB = im2 - np.mean(im2, axis=(0,1))
    sum_pix = lambda arr: np.sum(arr, axis=(0, 1))

    values = sum_pix((centeredA) * (centeredB)) / np.sqrt(sum_pix(np.power(centeredA, 2)) * sum_pix(np.power(centeredB, 2)))
    return np.mean(values)

