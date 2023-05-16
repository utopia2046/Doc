# Making Games with Python and Pygame

- [Book PDF & Source Code](https://inventwithpython.com/pygame/)
- [PyGame Documents](https://www.pygame.org/docs/)

## Installing Python and Pygame

``` console
python3 -m pip install -U pygame --user
```

## Pygame Basics

Game loop:
    events handling
    update game states
    rendering

### Common events

<https://www.pygame.org/docs/ref/event.html>

Event              | Parameters
-------------------|--------------------------------------------------
QUIT               | none
ACTIVEEVENT        | gain, state
KEYDOWN            | key, mod, unicode, scancode
KEYUP              | key, mod, unicode, scancode
MOUSEMOTION        | pos, rel, buttons, touch
MOUSEBUTTONUP      | pos, button, touch
MOUSEBUTTONDOWN    | pos, button, touch
VIDEORESIZE        | size, w, h
VIDEOEXPOSE        | none
USEREVENT          | code
AUDIODEVICEADDED   | which, iscapture (SDL backend >= 2.0.4)
AUDIODEVICEREMOVED | which, iscapture (SDL backend >= 2.0.4)
FINGERMOTION       | touch_id, finger_id, x, y, dx, dy
FINGERDOWN         | touch_id, finger_id, x, y, dx, dy
FINGERUP           | touch_id, finger_id, x, y, dx, dy
MOUSEWHEEL         | which, flipped, x, y, touch, precise_x, precise_y
MULTIGESTURE       | touch_id, x, y, pinched, rotated, num_fingers
TEXTEDITING        | text, start, length
TEXTINPUT          | text
DROPFILE           | file
DROPBEGIN          | (SDL backend >= 2.0.5)
DROPCOMPLETE       | (SDL backend >= 2.0.5)
DROPTEXT           | text (SDL backend >= 2.0.5)
MIDIIN             |
MIDIOUT            |
CLIPBOARDUPDATE    |

Game Surface: zero based (X, Y) coordinate system

### Draw methods

- pygame.draw.rect
- pygame.draw.polygon
- pygame.draw.circle
- pygame.draw.ellipse
- pygame.draw.arc
- pygame.draw.line: draw a straight line
- pygame.draw.lines: draw multiple contiguous straight line segments
- pygame.draw.aaline: draw a straight antialiased line
- pygame.draw.aalines

``` python
import pygame, sys
from pygame.locals import *

# Init
pygame.init()
FPS = 30 # frames per second setting
fpsClock = pygame.time.Clock()
# set up the window
DISPLAYSURF = pygame.display.set_mode((400, 300), 0, 32)
pygame.display.set_caption('Animation')

while True: # the main game loop
    for event in pygame.event.get():
        # handling events
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    # update states and render
    pygame.display.update()
    fpsClock.tick(FPS)
```

### Rendering images

- Load image: `pygame.image.load(path)`
- Render image: `DISPLAYSURF.blit(img, (x, y))`

### Draw texts

<https://www.pygame.org/docs/ref/font.html>

- Get system fonts: `pygame.font.get_fonts()`
- Create font object using system font: `pygame.sysfont.SysFont(name, size, bold=False, italic=False, constructor=None)`
- Create font object with file: `pygame.font.Font(path, size)`
- Create text rendering surface: `fontObj.render`
- Specify text location: `textSurfaceObj.get_rect()`
- Render text surface: `DISPLAYSURF.blit(textSurfaceObj, textRectObj)`

``` python
fontObj = pygame.font.Font(fontPath, 32)
textSurfaceObj = fontObj.render('Hellow World!', True, GREEN, BLUE)
textRectObj = textSurfaceObj.get_rect()
textRectObj.center = (200, 150)
DISPLAYSURF.blit(textSurfaceObj, textRectObj)
```

### Play sound

``` python
soundObj = pygame.mixer.Sound('beeps.wav')
soundObj.play()
# loop play background music
pygame.mixer.music.load(path)
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.stop()
```

## Memory Puzzle

``` python
def main():
    global FPSCLOCK, DISPLAYSURF # set FPSCLOCK and DISPLAYSURF as global since they're used by most functions and can't be changed locally
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    ...
    mainBoard = getRandomizedBoard() # mainBoard is a 2-D array saving (color, shape) tuples for each box
    revealedBoxes = generateRevealedBoxesData(False) # store which boxes are revealed
    firstSelection = None # stores the (x, y) of the first box clicked.
    ...
    while True: # main game loop, each iteration is a clock tick
        mouseClicked = False
        DISPLAYSURF.fill(BGCOLOR) # drawing the window
        drawBoard(mainBoard, revealedBoxes)
        for event in pygame.event.get(): # event handling loop
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION: # mouse move
                mousex, mousey = event.pos  # save mousex, mousey if mouse move or clicked
            elif event.type == MOUSEBUTTONUP: # mouse clicked
                mousex, mousey = event.pos
                mouseClicked = True
        boxx, boxy = getBoxAtPixel(mousex, mousey) # get box index from mouse x, y
        if boxx != None and boxy != None:
            # The mouse is currently over a box.
            if not revealedBoxes[boxx][boxy]:
                drawHighlightBox(boxx, boxy) # draw blue highlight box to notify user that current box is clickable
            if not revealedBoxes[boxx][boxy] and mouseClicked:
                revealBoxesAnimation(mainBoard, [(boxx, boxy)])
                revealedBoxes[boxx][boxy] = True # set the box as "revealed"
                if firstSelection == None: # the current box was the first box clicked
                    firstSelection = (boxx, boxy)
                else: # the current box was the second box clicked
                    # Check if there is a match between the two icons.
                    icon1shape, icon1color = getShapeAndColor(mainBoard, firstSelection[0], firstSelection[1])
                    icon2shape, icon2color = getShapeAndColor(mainBoard, boxx, boxy)

                    if icon1shape != icon2shape or icon1color != icon2color:
                        # Icons don't match. Re-cover up both selections.
                        coverBoxesAnimation(mainBoard, [(firstSelection[0], firstSelection[1]), (boxx, boxy)])
                        revealedBoxes[firstSelection[0]][firstSelection[1]] = False
                        revealedBoxes[boxx][boxy] = False
                    elif hasWon(revealedBoxes): # check if all pairs found
                        gameWonAnimation(mainBoard)
                        # Restart a new game
                        mainBoard = getRandomizedBoard()
                        revealedBoxes = generateRevealedBoxesData(False)
                        ...
                    firstSelection = None # reset firstSelection variable
        # Redraw the screen and wait a clock tick.
        pygame.display.update()
        FPSCLOCK.tick(FPS)

```

<!--
TODO:
unfinished
-->

## Slide Puzzle

## Simulate

## Wormy

## Tetromino

## Squirrel Eat Squirrel

## Star Pusher

## Four Extra Games
