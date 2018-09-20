from PIL import Image
class Hflip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        return (img, flipped)


    def __repr__(self):
        return self.__class__.__name__

def crop(img, row):
    pass

