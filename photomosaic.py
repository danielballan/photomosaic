import os
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import matplotlib
matplotlib.use('Agg')
import Image
import sqlite3
from utils import memo
import color_spaces as cs
from directory_walker import DirectoryWalker

def salient_colors(img, clusters=4, size=100):
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

def create_image_pool(image_dir, db_name='imagepool.db'):
    db = connect(os.path.join(image_dir, db_name))
    walker = DirectoryWalker(image_dir)
    for filename in walker:
        try:
            img = Image.open(filename)
        except IOError:
            print 'Cannot open %s as an image. Skipping it.' % filename
            continue
        w, h = img.size
        rgb_colors = salient_colors(img)
        lab_colors = map(cs.rgb2lab, rgb_colors)
        insert(filename, w, h, rgb_colors, lab_colors, db)
    db.commit()
    print_db(db)
    db.close()

def print_db(db):
    c = db.cursor()
    c.execute("SELECT * FROM Images")
    for row in c:
        print row 
    c.execute("SELECT * FROM Colors")
    for row in c:
        print row
    c.close()

def insert(filename, w, h, rgb, lab, db):
    c = db.cursor()
    try:
        c.execute("""INSERT INTO Images (usages, w, h, filename)
                     VALUES (?, ?, ?, ?)""",
                  (0, w, h, filename))
        image_id = c.lastrowid
        for i in xrange(len(rgb)):
            red, green, blue = map(int, rgb[i]) # np.uint8 confuses sqlite3
            L, a, b = lab[i] 
            rank = 1 + i
            c.execute("""INSERT INTO Colors (image_id, rank, 
                         L, a, b, red, green, blue) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (image_id, rank, L, a, b, red, green, blue))
    except sqlite3.IntegrityError:
        print "Image %s is already in the table. Skipping it." % filename
    c.close()
    

def connect(db_path):
    try:
        db = sqlite3.connect(db_path)
        create_tables(db)
    except IOError:
        print 'Cannot connect to SQLite database at %s' % db_path
        return
    return db

def create_tables(db):
    c = db.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS Images
                 (id INTEGER PRIMARY KEY,
                  usages INTEGER,
                  w INTEGER,
                  h INTEGER,
                  filename TEXT UNIQUE)""")
    c.execute("""CREATE TABLE IF NOT EXISTS Colors
                 (id INTEGER PRIMARY KEY,
                  image_id INTEGER,
                  rank INTEGER,
                  L REAL,
                  a REAL,
                  b REAL,
                  red INTEGER,
                  green INTEGER,
                  blue INTEGER)""")
    c.close()
    db.commit()

def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' ''.join(chr(c) for c in color).encode('hex')

def Lab_distance(lab1, lab2):
    """Compute distance in Lab."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    E = sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
    return E
    
def ab_distance(lab1, lab2):
    """Compute distance in the a-b plane, disregarding L."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return sqrt((a1-a2)**2 + (b1-b2)**2)

def characterize_image(img):
    colors = salient_colors(img)
    print [color_hex(c) for c in colors]

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

def fitness(img, target):
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
