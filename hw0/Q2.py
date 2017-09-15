from PIL import Image
import sys

fileName = sys.argv[1]
img = Image.open(fileName)

for i in range(img.size[0]):
    for j in range(img.size[1]):
        [r, g, b] = img.getpixel((i,j))
        newRGB = (int(r/2), int(g/2), int(b/2))
        img.putpixel((i,j), newRGB)

img.save('Q2.png')
