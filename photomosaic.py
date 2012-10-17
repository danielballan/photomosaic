from math import *

import matplotlib
matplotlib.use('Agg')
import Image
import pyccv # bitbucket.org/ynil/pyccv ; argmax.jp

import scipy
import scipy.misc
import scipy.cluster

def salient_colors(img, clusters=5, size=100):
    """Group the colors in an image into like clusters, and return a list
    of these colors in order of their abundance in the image."""
    img.thumbnail((size, size))
    imgarr = scipy.misc.fromimage(img)
    imgarr = imgarr.reshape(scipy.product(imgarr.shape[:2]), imgarr.shape[2])
    colors, dist = scipy.cluster.vq.kmeans(imgarr, clusters)
    vecs, dist = scipy.cluster.vq.vq(imgarr, colors)
    counts, bins = scipy.histogram(vecs, len(colors))
    ranked_colors = colors[(-counts).argsort()]
    return ranked_colors

def color_hex(color):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' ''.join(chr(c) for c in color).encode('hex')

def HSL(color):
    """Compute hue, saturation, lightness from [r, g, b].
    Reference: http://en.wikipedia.org/wiki/HSL_and_HSV#General_approach"""
    r, g, b = color
    alpha = (2*r - g - b)/2.
    beta = sqrt(3)*(g - b)/2.
    M = max(r, g, b)
    m = min(r, g, b)
    chroma = M -  m
    hue = 180./pi*atan2(beta, alpha) # approx good within 1.12 deg
    lightness = (M + m)/2.
    saturation = chroma/(1 - abs(2*lightness - 1)) if lightness else 0
    return hue, saturation, lightness

def characterize_image(img)
    colors = salient_colors(img)
    print [color_hex(c) for c in colors]
    h, s, l = HSL(color[0])

def find_match(img):
    # Rank images by their proximinity in h and s ONLY.
    # To match l, we can shrink the image in its spot, creating a
    # white margin. Almost always, light colors are in short supply, so
    # this adjustment goes in the right direction. SOME images might be too
    # bright, though, so we should require that l not be too much brighter
    # than the spot we're trying to match. Darker is fine though.
    # To avoid overusing the same images, introduce a probabilistic element
    # to punish images that have been used before.
    # Once the image is chosen, resize to create white padding if necessary.
    # To make the images appear jumbled, randomize each image's position within
    # its padding (if it is padded).
    pass

def grid(target_img):
    # Divide the target image into a regular mesh, and evaluate colors.
    # Merge rectangular regions that have the same color. (Use "labels"
    # from scipy?) Now it's a list of spots, not a regular array. Find
    # a match for each spot. 
    # Alternative: Instead of merging based on color, evaluate the local
    # spatial wavelength. Merge regions with slow variation.
    pass 

img = Image.open('images/earth.jpg')
print salient_colors(img)
