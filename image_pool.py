from directory_walker import DirectoryWalker
from progress_bar import progress_bar
import Image
from image_functions import *
import logging

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

class ImagePool(dict):
    def __init__(self):
        None

    def add_directory(self, image_dir):
        walker = DirectoryWalker(image_dir)
        file_count = len(list(walker)) # stupid but needed but progress bar
        pbar = progress_bar(file_count, "Analyzing images and building db")

        for filename in walker:
            if filename in self:
                logger.warning("Image %s is already in the table. Skipping it."%filename)
                pbar.next()
                continue
            try:
                img = Image.open(filename)
            except IOError:
                logger.warning("Cannot open %s as an image. Skipping it.",
                               filename)
                pbar.next()
                continue
            if img.mode != 'RGB':
                logger.warning("RGB images only. Skipping %s.", filename)
                pbar.next()
                continue
            w, h = img.size
            try:
                regions = split_quadrants(img)
                rgb = map(dominant_color, regions) 
                lab = map(cs.rgb2lab, rgb)
            except Exception as e:
                logger.warning("Unknown problem analyzing %s. (%s) Skipping it.",
                               filename, str(e))
                continue
            self.insert(filename, w, h, rgb, lab)
            pbar.next()
        logger.info('Collection %s built with %d images'%(self.db_name, len(self)))
