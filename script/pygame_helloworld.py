import pygame, sys, os
from pygame.locals import *

# https://www.pygame.org/docs/ref/display.html
pygame.init()
# set_mode(size=(0, 0), flags=0, depth=0, display=0, vsync=0) -> Surface
DISPLAYSURF = pygame.display.set_mode((400, 300), 0, 32)
# create a transparent surface
# anotherSurface = DISPLAYSURF.convert_alpha()
pygame.display.set_caption('Hello World!')

# set up the colors
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)

# draw on the surface object
DISPLAYSURF.fill(WHITE)
#pygame.draw.polygon(DISPLAYSURF, GREEN, ((146, 0), (291, 106), (236, 277), (56, 277), (0, 106)))
#pygame.draw.line(DISPLAYSURF, BLUE, (60, 60), (120, 60), 4)
#pygame.draw.line(DISPLAYSURF, BLUE, (120, 60), (60, 120))
#pygame.draw.line(DISPLAYSURF, BLUE, (60, 120), (120, 120), 4)
#pygame.draw.circle(DISPLAYSURF, BLUE, (300, 50), 20, 0)
#pygame.draw.ellipse(DISPLAYSURF, RED, (300, 200, 40, 80), 1)
#pygame.draw.rect(DISPLAYSURF, RED, (200, 150, 100, 50))

# update disply surface at pixel level
pixObj = pygame.PixelArray(DISPLAYSURF)
pixObj[380][280] = BLACK
pixObj[382][282] = BLACK
pixObj[384][284] = BLACK
pixObj[386][286] = BLACK
pixObj[388][288] = BLACK
del pixObj  # tell pygame to unblock surface

# draw text
root = os.getcwd()
fontPath = os.path.join(root, 'res\\freesansbold.ttf')
fontObj = pygame.font.Font(fontPath, 32)
textSurfaceObj = fontObj.render('Hellow World!', True, GREEN, BLUE)
textRectObj = textSurfaceObj.get_rect()
textRectObj.center = (200, 150)
DISPLAYSURF.blit(textSurfaceObj, textRectObj)

# play sound
soundPath = os.path.join(root, 'res\\beep1.ogg')
soundObj = pygame.mixer.Sound(soundPath)
soundObj.play()
import time
time.sleep(1) # wait and let the sound play for 1 second
soundObj.stop()

# main game loop
while True:
    for event in pygame.event.get():
        # https://www.pygame.org/docs/ref/event.html
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
