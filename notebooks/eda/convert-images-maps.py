## HTMLS rendered in notebooks/eda/anomalies-eagle-s-vezhen-20260617.py have screenshots taken of them in high quality.

from pathlib import Path
from PIL import Image

img_path = Path("../images/map-yipeng3.png")
tiff_path = Path("../images/Figure_1.tif")

with Image.open(img_path) as image:
    image.save(tiff_path, format="TIFF", dpi=(1000, 1000), compression='tiff_lzw')

img_path = Path("../images/map-eagles-vezhen.png")
tiff_path = Path("../images/Figure_3.tif")

with Image.open(img_path) as image:
    image.save(tiff_path, format="TIFF", dpi=(1000, 1000), compression='tiff_lzw')
