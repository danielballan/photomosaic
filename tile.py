
from memo import memo
import Image
import ImageFilter
import logging
from image_functions import *

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

@memo
def open_tile(filename, temp_size=(100,100)):
    """This memoized function only opens each image once."""
    im = Image.open(filename)
    im.thumbnail(temp_size, Image.ANTIALIAS) # Resize to fit within temp_size without cropping.
    return im


class Tile(object):
    """Tile wraps the Image class, so all methods that apply to images (show,
    save, crop, size, ...) apply to Tiles. Tiles also store contextual
    information that is used to reassembled them in the end."""
    def __init__(self, img, x, y, mask=None, ancestry=[], ancestor_size=None):
        self._img = img
        self.x = x
        self.y = y
        self._mask = mask.convert("L") if mask else None
        self._blank = None # meaning undetermined (so far)
        self._ancestry = ancestry
        self._depth = len(self._ancestry)
        if ancestor_size:
            self._ancestor_size = ancestor_size
        else:
            self._ancestor_size = self.size

    def crop(self, *args):
        if self._mask: self._mask.crop(*args)
        return self._img.crop(*args)

    def resize(self, *args):
        if self._mask: self._mask.resize(*args)
        return self._img.resize(*args)

    def __getattr__(self, key):
        if key == '_img':
            raise AttributeError()
        return getattr(self._img, key)

    def pos(self):
        return self.x, self.y 

    def avg_color(self):
        t = [0]*3
        for rgb in self._rgb:
            for i, c in enumerate(rgb):
                t[i] += c
        return [a/len(self._rgb) for a in t] 

    @property
    def ancestry(self):
        return self._ancestry

    @property
    def depth(self):
        return self._depth

    @property
    def ancestor_size(self):
        return self._ancestor_size

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    @property
    def lab(self):
        return self._lab

    @lab.setter
    def lab(self, value):
        self._lab = value

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        self._match = value # sqlite Row object
        try:
            self._match_img = open_tile(self._match['filename'],
                (2*self._ancestor_size[1], 2*self.ancestor_size[0]))
                # Reversed on purpose, for thumbnail. Largest possible size
                # we could want later.
        except IOError:
            logger.error("The filename specified in the database as "
                         "cannot be found. Check: %s", self._match['filename'])

    @property
    def match_img(self):
        return self._match_img

    @property
    def blank(self):
        return self._blank
        
    def analyze(self):
        """"Determine dominant colors of target tile, and save that information"""
        if self.blank:
            return
        regions = split_quadrants(self)
        self.rgb = map(dominant_color, regions) 
        self.lab = map(cs.rgb2lab, self.rgb)

    def determine_blankness(self, min_depth=1):
        """Decide whether this tile is blank. Where the mask is grey, tiles
        and blanked probabilitisically. The kwarg min_depth limits this
        scattered behavior to small tiles."""
        if not self._mask: # no mask
            self._blank = False
            return
        brightest_pixel = self._mask.getextrema()[1]
        if brightest_pixel == 0: # black mask 
            self._blank = True
        elif brightest_pixel == 255: # white mask
            self._blank = False
        elif self._depth < min_depth: # gray mask -- big tile
            self._blank = True
        elif 255*np.random.rand() > brightest_pixel: # small tile
            self._blank = True
        else:
            self._blank = False
        return

    def straddles_mask_edge(self):
        """A tile straddles an edge if it contains PURE white (255) and some
        nonwhite. A tile that contains varying shades of gray does not
        straddle an edge."""
        if not self._mask:
            return False
        darkest_pixel, brightest_pixel = self._mask.getextrema()
        if brightest_pixel != 255:
            return False
        if darkest_pixel == 255:
            return False
        return True
 
    def dynamic_range(self):
        """What is the dynamic range in this image? Return the
        average dynamic range over RGB channels. Blur the image
        first to smooth away outliers."""
        return sum(map(lambda (x, y): y - x, 
                       self._img.filter(ImageFilter.BLUR).getextrema()))//3

    def procreate(self):
        """Divide image into quadrants, make each into a child tile,
        and return them all in a list.""" 
        width = self._img.size[0] // 2
        height = self._img.size[1] // 2
        children = []
        for y in [0, 1]:
            for x in [0, 1]:
                tile_img = self._img.crop((x*width, y*height,
                                    (x + 1)*width, (y + 1)*height))
                if self._mask:
                    mask_img = self._mask.crop((x*width, y*height,
                                         (x + 1)*width, (y + 1)*height))
                else:
                    mask_img = None
                child = Tile(tile_img, self.x, self.y,
                             mask=mask_img,
                             ancestry=self._ancestry + [(x, y)],
                             ancestor_size=self._ancestor_size)
                children.append(child)
        return children
