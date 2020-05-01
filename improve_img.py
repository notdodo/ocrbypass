from PIL import Image
from os.path import basename
import sys

original = Image.open(sys.argv[1])
improved = Image.new("RGB", (200, 70), "white")
improved_pixels = improved.load()
black = (0, 0, 0)
white = (255, 255, 255)

# Get diffs from images and save to a new blank PNG
for x in range(0, original.size[0]):
    for y in range(0, original.size[1]):
        or_r, or_g, or_b = original.getpixel((x, y))
        if any(p > 230 for p in [or_r, or_g, or_b]):
            # if or_r + or_g + or_b > 510:
            improved_pixels[x, y] = white
        else:
            improved_pixels[x, y] = black

improved.save("./improved/" + basename(sys.argv[1]), "png")
