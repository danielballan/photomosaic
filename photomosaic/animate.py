import photomosaic as pm
import Image
import random
import operator
from sql_image_pool import SqlImagePool
from gui_utils import MosaicGUI
from cli import *

def color_sort(tile):
    if tile.blank:
        return -1
    return tile.avg_color()

def random_sort(tile):
    return random.random()

def no_sort(tile):
    return 1
    
def size_sort(tile):
    return -tile.size[0]

analyze_sort = no_sort
match_sort = color_sort

locs = [[0,0], [1,0], [0,1], [1,1]]

parser = args_parser()
args = parser.parse_args()

pool = get_database(args)

p = pm.Photomosaic(args.infile, pool, tuning=args.tune, mask=args.mask)
W,H = p.orig_img.size
Hx = 0
Wx = W
gui = MosaicGUI((W+Wx,H+Hx))
gui.rectangle((255,255,255), (0,0,W+Wx,H+Hx))
gui.img(p.orig_img, (0,0))
gui.draw()

p.partition_tiles(args.dimensions, depth=args.recursion_level, analyze=False)

for tile in sorted(p.tiles, key=no_sort):
    if tile.blank:
        continue
    tx,ty = tile.get_position(tile.size)
    w,h = tile.size
    gui.rectangle((0,0,255), (tx+Wx,ty+Hx,w,h), 1)
    gui.draw()

for tile in sorted(p.tiles, key=size_sort):
    if tile.blank:
        continue

    tile.analyze()
    tx,ty = tile.get_position(tile.size)
    w,h = tile.size
    for (x,y), color in zip(locs, tile.rgb):
        rect = (Wx+tx + w*x/2,Hx+ty+h*y/2,w/2,h/2)
        gui.rectangle(color, rect)
    gui.rectangle((0,0,0), (tx+Wx,ty+Hx,w,h), 1)
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

