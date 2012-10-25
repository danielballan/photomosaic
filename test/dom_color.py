import photomosaic as pm
import pygame
import Image
import sys
pygame.init()
img = Image.open(sys.argv[1])

regions = pm.split_quadrants(img)

rgb = map(pm.dominant_color, regions) 
locs = [[0,0], [1,0], [0,1], [1,1]]
print rgb
w,h = img.size
screen = pygame.display.set_mode((w,h*2))

pimg = pygame.image.load(sys.argv[1])
prect = pimg.get_rect()

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    screen.fill((1,1,1))
    screen.blit(pimg, prect)
    for (x,y), color in zip(locs, rgb):
        pygame.draw.rect(screen, color, (w*x/2,h+h*y/2,w/2,h/2))
    pygame.display.flip()
