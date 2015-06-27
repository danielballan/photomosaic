import photomosaic as pm
import Image
import sys
import random
import operator
from sql_image_pool import SqlImagePool
from gui_utils import MosaicGUI

DEPTH = 2

def color_sort(tile):
    return tile.avg_color()

def random_sort(tile):
    return random.random()

def no_sort(tile):
    return 1

analyze_sort = no_sort
match_sort = color_sort

locs = [[0,0], [1,0], [0,1], [1,1]]

infile = sys.argv[1]
database = sys.argv[2]
tune = '-tune' in sys.argv

pool = SqlImagePool(database)

p = pm.Photomosaic(infile, pool, tuning=tune)
W,H = p.orig_img.size
Hx = 0
Wx = W
gui = MosaicGUI((W+Wx,H+Hx))

gui.img(p.orig_img, (0,0))
gui.draw()

p.partition_tiles(10, depth=DEPTH)

for tile in sorted(p.tiles, key=analyze_sort):
    tx,ty = tile.get_position(tile.size)
    w,h = tile.size
    for (x,y), color in zip(locs, tile.rgb):
        rect = (Wx+tx + w*x/2,Hx+ty+h*y/2,w/2,h/2)
        gui.rectangle(color, rect)
        gui.rectangle((0,0,0), rect, 1)
        gui.draw()

try:
    for tile in sorted(p.tiles, key=match_sort):
        if tile.blank:
            continue
        p.match_one(tile)    
        tx,ty = tile.get_position(tile.size)
        w,h = tile.size
        gui.scaled_img(tile.match[4], tx+Wx, ty+Hx, w, h)
        gui.draw()

finally:
    pool.close()
gui.wait_for_close()

