import scipy
import scipy.misc
from scipy.cluster import vq
import color_spaces as cs
import numpy as np
import Image

import logging

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def open(target_filename):
    "Just a wrapper for Image.open from PIL"
    try:
        return Image.open(target_filename)
    except IOError:
        logger.warning("Cannot open %s as an image.", target_filename)
        return

def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' + ''.join(chr(c) for c in rgb).encode('hex')

def split_regions(img, split_dim):
    """Split an image into subregions.
    Use split_dim=2 or (2,2) or (2,3) etc.
    Return a flat list of images."""
    if isinstance(split_dim, int):
        rows = columns = split_dim
    else:
        columns, rows = split_dim
    r_size = img.size[0] // columns, img.size[1] // rows
    # regions = [[None for c in range(columns)] for r in range(rows)]
    regions = columns*rows*[None]
    for y in range(rows):
        for x in range(columns):
            region = img.crop((x*r_size[0], 
                             y*r_size[1],
                             (x + 1)*r_size[0], 
                             (y + 1)*r_size[1]))
            # regions[y][x] = region ## for nested output
            regions[y*columns + x] = region
    return regions
    
def split_quadrants(img):
    """Convenience function: calls split_regions(img, 2). Returns
    a flat 4-element list: top-left, top-right, bottom-left, bottom-right."""
    if img.size[0] & 1 or img.size[1] & 1:
        logger.debug("I am quartering an image with odd dimensions.")
    return split_regions(img, 2)

def dominant_color(img, clusters=5, size=50):
    """Group the colors in an image into like clusters, and return
    the central value of the largest cluster -- the dominant color."""
    assert img.mode == 'RGB', 'RGB images only!'
    img.thumbnail((size, size))
    imgarr = scipy.misc.fromimage(img)
    imgarr = imgarr.reshape(scipy.product(imgarr.shape[:2]), imgarr.shape[2])
    colors, dist = vq.kmeans(imgarr, clusters)
    vecs, dist = vq.vq(imgarr, colors)
    counts, bins = scipy.histogram(vecs, len(colors))
    dominant_color = colors[counts.argmax()]
    return map(int, dominant_color) # Avoid returning np.uint8 type.

def plot_histograms(hist, title=''):
    "Plot an RGB histogram given as a dictionary with channel keys."
    import matplotlib.pyplot as plt
    fig, (red, green, blue) = plt.subplots(3, sharex=True, sharey=True)
    domain = range(0, 256)
    red.fill_between(domain, hist['red'],
                     facecolor='red')
    green.fill_between(domain, 0, hist['green'],
                       facecolor='green')
    blue.fill_between(domain, 0, hist['blue'],
                      facecolor='blue')
    red.set_xlim(0,256)
    red.set_ylim(ymin=0)
    red.set_title(title)
    fig.show()

def img_histogram(img, mask=None):
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, img.split()))
    hist= {}
    for ch in keys:
        if mask:
            h = channels[ch].histogram(mask.convert("1"))
        else:
            h = channels[ch].histogram()
        normalized_h = [256./sum(h)*v for v in h]
        hist[ch] = normalized_h
    return hist

def compute_palette(hist):
    """A palette maps a channel into the space of available colors, gleaned
    from a histogram of those colors."""
    # Integrate a histogram and round down.
    palette = {}
    for ch in ['red', 'green', 'blue']:
        integrals = np.cumsum(hist[ch])
        blocky_integrals = np.floor(integrals + 0.01).astype(int)
        bars = np.ediff1d(blocky_integrals,to_begin=blocky_integrals[0])
        p = [[color]*freq for color, freq in enumerate(bars.tolist())]
        p = [c for sublist in p for c in sublist]
        assert len(p) == 256, "Palette should have 256 entries."
        palette[ch] = p
    return palette

def adjust_levels(target_img, from_palette, to_palette):
    """Transform the colors of an image to match the color palette of
    another image."""
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, target_img.split()))
    f, g = from_palette, to_palette # compact notation
    func = {} # function to transform color at each pixel
    for ch in keys:
        def j(x):
           while True:
               try:
                   inv_f = f[ch].index(x)
                   break
               except ValueError:
                   if x < 255:
                       x += 1
                       continue 
                   else:
                       inv_f = 255
                       break
           return to_palette[ch][inv_f]
        func[ch] = j 
    adjusted_channels = [Image.eval(channels[ch], func[ch]) for ch in keys]
    return Image.merge('RGB', adjusted_channels)

def crop_to_fit(img, tile_size):
    "Return a copy of img cropped to precisely fill the dimesions tile_size."
    img_w, img_h = img.size
    tile_w, tile_h = tile_size
    img_aspect = img_w/img_h
    tile_aspect = tile_w/tile_h
    if img_aspect > tile_aspect:
        # It's too wide.
        crop_h = img_h
        crop_w = int(round(crop_h*tile_aspect))
        x_offset = int((img_w - crop_w)/2)
        y_offset = 0
    else:
        # It's too tall.
        crop_w = img_w
        crop_h = int(round(crop_w/tile_aspect))
        x_offset = 0
        y_offset = int((img_h - crop_h)/2)
    img = img.crop((x_offset,
                    y_offset,
                    x_offset + crop_w,
                    y_offset + crop_h))
    img = img.resize((tile_w, tile_h), Image.ANTIALIAS)
    return img
