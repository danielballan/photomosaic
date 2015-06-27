import pygame
import sys

def topygame(img):
    return pygame.image.frombuffer(img.tostring(), img.size, "RGB")

def draw(img, box):
    screen.blit(topygame(img), box)

def draw_scaled(filename, x,y,w,h):
    screen.blit(pygame.transform.scale(pygame.image.load(filename), (w, h)), (x,y))

class MosaicGUI:
    def __init__(self, size):
        pygame.init()
        self.screen = pygame.display.set_mode(size)

    def draw(self):
        pygame.display.flip()
        
    def rectangle(self, color, coords, width=0):
        pygame.draw.rect(self.screen, color, coords, width)    
        
    def img(self, image, box):
        self.screen.blit(topygame(image), box)
        
    def scaled_img(self, filename, x,y,w,h):
        self.screen.blit(pygame.transform.scale(pygame.image.load(filename), (w, h)), (x,y))

    def wait_for_close(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
