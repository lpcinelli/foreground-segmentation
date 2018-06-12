from PIL import ImageFile  # To solve load problems

# Valid images extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm'
]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    '''
    Test if current file is a valid image.

    @param filename Input file name.

    @return True if file is an image, else False.
    '''
    return filename.endswith(tuple(IMG_EXTENSIONS))
