import matplotlib
matplotlib.use('Agg')
import Image
import pyccv # bitbucket.org/ynil/pyccv ; argmax.jp

import scipy
import scipy.misc
import scipy.cluster

def freq_colors(img, clusters=5, size=240):
    img.thumbnail((size, size))
    ar = scipy.misc.fromimage(img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2])
    codes, dist = scipy.cluster.vq.kmeans(ar, clusters)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)
    counts, bins = scipy.histogram(vecs, len(codes))
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    color = ''.join(chr(c) for c in peak).encode('hex')
    return color

img = Image.open('images/field.jpg')
print freq_colors(img)
