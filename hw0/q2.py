import sys
from PIL import Image

imgname = sys.argv[1]

src_im = Image.open(imgname)
angle = 180

dst_im = src_im.rotate( angle )
dst_im.save("ans2.png")
