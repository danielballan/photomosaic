import photomosaic as pm
import pygame
import Image
import sys
import random
import operator
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

orig_img = pm.open( infile )
orig_img.load()
W,H = orig_img.size
Hx = 0
Wx = W
screen = pygame.display.set_mode((W+Wx,H+Hx))
draw(orig_img, (0,0))
pygame.display.flip()

if tune:
    img = pm.tune(orig_img, database) # Adjust colors levels to what's availabe in the pool.
else:
    img = orig_img

tiles = pm.partition(img, (10, 10), depth=DEPTH)

for tile in sorted(tiles, key=analyze_sort):
    pm.analyze_one(tile)
    tx,ty = pm.tile_position(tile)
    w,h = tile.size
    for (x,y), color in zip(locs, tile.rgb):
        rect = (Wx+tx + w*x/2,Hx+ty+h*y/2,w/2,h/2)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, 1)
        pygame.display.flip()

db = pm.connect(database)
try:
    pm.reset_usage(db)
    for tile in sorted(tiles, key=match_sort):
        if tile.blank:
            continue
        tile.match = pm.choose_match(tile.lab, db)
        tx,ty = pm.tile_position(tile)
        w,h = tile.size
        draw_scaled(tile.match[4], tx+Wx, ty+Hx, w, h)
        pygame.display.flip()
        
finally:
    db.close()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

