import photomosaic as pm
import pygame
import Image
import sys
import random
import operator
from sql_image_pool import SqlImagePool
pygame.init()

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

def topygame(img):
    return pygame.image.frombuffer(img.tostring(), img.size, "RGB")

def draw(img, box):
    screen.blit(topygame(img), box)

def draw_scaled(filename, x,y,w,h):
    screen.blit(pygame.transform.scale(pygame.image.load(filename), (w, h)), (x,y))

infile = sys.argv[1]
database = sys.argv[2]
tune = '-tune' in sys.argv

pool = SqlImagePool(database)

p = pm.Photomosaic(infile, pool, tuning=tune)
W,H = p.orig_img.size
Hx = 0
Wx = W
screen = pygame.display.set_mode((W+Wx,H+Hx))
draw(p.orig_img, (0,0))
pygame.display.flip()

p.partition_tiles(10, depth=DEPTH)

for tile in sorted(p.tiles, key=analyze_sort):
    tx,ty = tile.get_position(tile.size)
    w,h = tile.size
    for (x,y), color in zip(locs, tile.rgb):
        rect = (Wx+tx + w*x/2,Hx+ty+h*y/2,w/2,h/2)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, 1)
        pygame.display.flip()

try:
    for tile in sorted(p.tiles, key=match_sort):
        if tile.blank:
            continue
        p.match_one(tile)    
        tx,ty = tile.get_position(tile.size)
        w,h = tile.size
        draw_scaled(tile.match[4], tx+Wx, ty+Hx, w, h)
        pygame.display.flip()

finally:
    pool.close()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

