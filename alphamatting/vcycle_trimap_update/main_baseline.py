import numpy as np
import scipy.sparse
from PIL import Image
import numpy as np
from pymatting import cg, vcycle, cf_laplacian, ProgressCallback
import time

def main():
    print("loading images")

    size = (34, 22)
    size = (680, 440)
    image = np.array(Image.open("images/lemur.png").convert("RGB").resize(size, Image.BOX)) / 255.0
    trimap = np.array(Image.open("images/lemur_trimap.png").convert("L").resize(size, Image.NEAREST)) / 255.0

    is_fg = trimap == 1.0
    is_bg = trimap == 0.0
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    b = 100.0 * is_fg.flatten()
    c = 100.0 * is_known.flatten()

    shape = trimap.shape
    h, w = shape

    L = cf_laplacian(image)
    C = scipy.sparse.diags(c)
    A = L + C

    M = vcycle(A, (h, w))

    x = cg(A, b, M=M, callback=ProgressCallback())

    print("\nbaseline:")
    print("iteration      69 - 2.690681e-03 (0.00269068076571873623)")

    alpha = np.clip(x, 0, 1).reshape(h, w)

    import matplotlib.pyplot as plt
    for i, img in enumerate([image, trimap, alpha]):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
