from PIL import Image
import sys

img = Image.open(sys.argv[1])

for i in range(img.size[0]):
    for j in range(img.size[1]):
        [r, g, b] = img.getpixel((i,j))
        newRGB = (r//2, g//2, b//2)
        img.putpixel((i,j), newRGB)

img.save('Q2.png')
