from PIL import ImageFile, Image  # To solve load problems
from skimage import io

from torchvision.datasets.folder import default_loader as torch_default_loader

# Valid images extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif', '.TIF'
]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    '''
    Test if current file is a valid image.

    @param filename Input file name.

    @return True if file is an image, else False.
    '''
    return filename.endswith(tuple(IMG_EXTENSIONS))


def default_loader(path):
    try:
        return torch_default_loader(path)
    except OSError:
        import numpy as np
        if path.endswith(tuple(['.tif', '.TIF'])):
            data = (io.imread(path).astype(np.float32) / (2**8 + 1)).astype(
                np.uint8)
            return Image.fromarray(data)
        raise
