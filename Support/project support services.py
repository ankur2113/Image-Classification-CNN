from PIL import Image
import os
import sys

path = ("C:/Users/15593/OneDrive/Documents/IT298/data1/Dataset/test/freshoranges/")
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((100, 100), Image.ANTIALIAS)
            imResize.save(f + ' resized.png', 'PNG', quality=90)

resize()