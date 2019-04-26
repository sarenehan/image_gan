import os
from os.path import join
from PIL import Image
import numpy as np
from scipy.ndimage import imread


directory = '/Users/stewart/projects/image_gan/data/test2014_2'
image_size = (64, 64)


def get_image_filenames():
    return [join(directory, f) for f in os.listdir(directory)]


def resize_and_greyscale_images():

    for filename in get_image_filenames():
        im = Image.open(filename)
        im = im.resize(image_size).convert('L')
        im.save(filename, "JPEG")


def show_image(image):
    Image.fromarray(image, mode='L').show()


def load_images():
    return [imread(filename) for filename in get_image_filenames()]


def load_training_data():
    return np.array([im.ravel() for im in load_images()])
