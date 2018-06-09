import sys # Read command line argument
from PIL import Image # Load/Save images
from image_recovery import *


def corrupt_and_restore(img, ratio):
    perturbed = corrupt(img, ratio)
    restored = recover(perturbed, 500)

    print('ratio=', ratio, 'similarity(orig, perturbed)=', similarity(perturbed, img))
    print('ratio=', ratio, 'similarity(orig, recovered)=', similarity(restored, img))

    Image.fromarray(perturbed, mode='RGB').save(f'perturbed-{ratio}.png')
    Image.fromarray(restored, mode='RGB').save(f'restored-{ratio}.png')


img = np.asarray(Image.open(sys.argv[1]), dtype=np.uint8)

corrupt_and_restore(img, 0.3)
corrupt_and_restore(img, 0.2)
corrupt_and_restore(img, 0.1)
corrupt_and_restore(img, 0.05)

