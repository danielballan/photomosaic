import numpy as np

def rgb2hsl(rgb):
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

def rgb2xyz_wikipedia(rgb):
    """Compute X, Y, Z from [r, g, b]. This color space is an intermediate
    step is converting RGB to LAB.
    Ref: http://en.wikipedia.org/wiki/CIE_1931_color_space"""
    matrix = 1/0.17697*np.array([[0.49, 0.31, 0.20], 
                                 [0.17697, 0.81240, 0.01063],
                                 [0.00, 0.01, 0.99]])
    xyz = np.dot(matrix, np.array(rgb)/255.)
    return xyz

def _fxyz(t):
    """Utility function used by rgb2xyz."""
    t *= 1/255.
    if t > 0.04045:
        return 100*((t + 0.055)/1.055)**2.4
    else:
        return 100*t/12.92

def rgb2xyz(rgb):
    """Compute X, Y, Z from [r, g, b].
    Ref: http://www.easyrgb.com/index.php?X=MATH&H=02#text2"""
    r, g, b = map(_fxyz, rgb) 
    matrix = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
    x, y, z = np.dot(matrix, np.array([r, g, b]))
    return x, y, z

def _f(t):
    """Utility function used in xyz2CIE_Lab().
    Ref: http://en.wikipedia.org/wiki/Lab_color_space
         #CIELAB-CIEXYZ_conversions"""
    if t > 0.00885645:
        return t**(0.333333)
    else:
         return 7.787037*t + 0.137931

def xyz2CIE_Lab(xyz):
    """Compute L, a, b from [x, y, z]. In the CIE-Lab color space, Cartesnian
    distance between points is a proxy for their perceptual difference.
    Ref: http://en.wikipedia.org/wiki/Lab_color_space.
            #CIELAB-CIEXYZ_conversions
    Ref: http://www.easyrgb.com/index.php?X=MATH&H=07#text7"""
    # Observer at 2 degrees, Illuminant = D65
    xn =  95.047
    yn = 100.000
    zn = 108.883
    x, y, z = xyz
    fx, fy, fz = _f(x/xn), _f(y/yn), _f(z/zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return L, a, b

def rgb2CIE_Lab(rgb):
    return xyz2CIE_Lab(rgb2xyz(rgb))

rgb2lab = rgb2CIE_Lab
