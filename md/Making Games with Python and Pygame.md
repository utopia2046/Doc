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

## Slide Puzzle

``` python
def main():
    ...
    # a new puzzle is generated by 80 random moves on a getStartingBoard (also SOLVEBOARD)
    mainBoard, solutionSeq = generateNewPuzzle(80) # mainBoard is a 2D array saving tile ids and None for the empty slot
    SOLVEDBOARD = getStartingBoard() # a solved board is the same as the board in a start state.
    allMoves = [] # list of moves made from the solved configuration
    while True: # main game loop
        slideTo = None # the direction, if any, a tile should slide
        if mainBoard == SOLVEDBOARD:
            msg = 'Solved!'
        for event in pygame.event.get(): # event handling loop
            if event.type == MOUSEBUTTONUP: # get tile location and blank spot to decide move direction
                spotx, spoty = getSpotClicked(mainBoard, event.pos[0], event.pos[1])
                if (spotx, spoty) != (None, None):
                    # check if the clicked tile was next to the blank spot
                    blankx, blanky = getBlankPosition(mainBoard)
                    if spotx == blankx + 1 and spoty == blanky:
                        slideTo = LEFT
                    elif spotx == blankx - 1 and spoty == blanky:
                        slideTo = RIGHT
                    elif spotx == blankx and spoty == blanky + 1:
                        slideTo = UP
                    elif spotx == blankx and spoty == blanky - 1:
                        slideTo = DOWN
            elif event.type == KEYUP:
                # check if the user pressed a key to slide a tile
                if event.key in (K_LEFT, K_a) and isValidMove(mainBoard, LEFT):
                    slideTo = LEFT
                elif event.key in (K_RIGHT, K_d) and isValidMove(mainBoard, RIGHT):
                    slideTo = RIGHT
                elif event.key in (K_UP, K_w) and isValidMove(mainBoard, UP):
                    slideTo = UP
                elif event.key in (K_DOWN, K_s) and isValidMove(mainBoard, DOWN):
                    slideTo = DOWN
        if slideTo:
            slideAnimation(mainBoard, slideTo, 'Click tile or press arrow keys to slide.', 8) # show slide on screen
            makeMove(mainBoard, slideTo)
            allMoves.append(slideTo) # record the slide
        pygame.display.update()
        FPSCLOCK.tick(FPS)
```

Notice that draw text is tiresome in pygame, so we use this function to draw text

``` python
def makeText(text, color, bgcolor, top, left):
    # create the Surface and Rect objects for some text.
    textSurf = BASICFONT.render(text, True, color, bgcolor)
    textRect = textSurf.get_rect()
    textRect.topleft = (top, left)
    return (textSurf, textRect)
```

In PyGame, we need to make a copy of current display surface and call blit to render animation

``` python
def slideAnimation(board, direction, message, animationSpeed):
    ...
    # prepare the base surface
    drawBoard(board, message)
    baseSurf = DISPLAYSURF.copy()
    # draw a blank space over the moving tile on the baseSurf Surface.
    moveLeft, moveTop = getLeftTopOfTile(movex, movey)
    pygame.draw.rect(baseSurf, BGCOLOR, (moveLeft, moveTop, TILESIZE, TILESIZE))

    for i in range(0, TILESIZE, animationSpeed):
        # animate the tile sliding over
        checkForQuit()
        DISPLAYSURF.blit(baseSurf, (0, 0))
        if direction == UP:
            drawTile(movex, movey, board[movex][movey], 0, -i)
        if direction == DOWN:
            drawTile(movex, movey, board[movex][movey], 0, i)
        if direction == LEFT:
            drawTile(movex, movey, board[movex][movey], -i, 0)
        if direction == RIGHT:
            drawTile(movex, movey, board[movex][movey], i, 0)

        pygame.display.update()
        FPSCLOCK.tick(FPS)
```

## Simulate

``` python
def main():
    # Initialize some variables for a new game
    pattern = [] # stores the pattern of colors
    currentStep = 0 # the color the player must push next
    lastClickTime = 0 # timestamp of the player's last button push
    score = 0
    # when False, the pattern is playing. when True, waiting for the player to click a colored button:
    waitingForInput = False

    while True: # main game loop
        clickedButton = None # button that was clicked (set to YELLOW, RED, GREEN, or BLUE)
        ...
        checkForQuit()
        for event in pygame.event.get(): # event handling loop
            if event.type == MOUSEBUTTONUP:
                mousex, mousey = event.pos
                clickedButton = getButtonClicked(mousex, mousey)

        if not waitingForInput:
            # play the pattern
            pygame.display.update()
            pygame.time.wait(1000)
            pattern.append(random.choice((YELLOW, BLUE, RED, GREEN)))
            for button in pattern:
                flashButtonAnimation(button)
                pygame.time.wait(FLASHDELAY)
            waitingForInput = True
        else:
            # wait for the player to enter buttons
            if clickedButton and clickedButton == pattern[currentStep]:
                # pushed the correct button
                flashButtonAnimation(clickedButton)
                currentStep += 1
                lastClickTime = time.time()

                if currentStep == len(pattern):
                    # pushed the last button in the pattern
                    changeBackgroundAnimation()
                    score += 1
                    waitingForInput = False
                    currentStep = 0 # reset back to first step

            elif (clickedButton and clickedButton != pattern[currentStep]) or (currentStep != 0 and time.time() - TIMEOUT > lastClickTime):
                # pushed the incorrect button, or has timed out
                gameOverAnimation()
                # reset the variables for a new game:
                pattern = []
                currentStep = 0
                waitingForInput = False
                score = 0
                pygame.time.wait(1000)
                changeBackgroundAnimation()

        pygame.display.update()
        FPSCLOCK.tick(FPS)
```

### Zen of Python

Try `import this` in IPython, will see the following code:

The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're [Dutch](#dutch).
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

#### Dutch

The inventor of Python is a Dutch programmer named Guido van Rossum. See <https://ell.stackexchange.com/questions/85517/zen-of-python-meaning-of-unless-youre-dutch>

## Wormy

``` python
def main():
    ...
    showStartScreen()
    while True:
        runGame()
        showGameOverScreen()

def runGame():
    # Set a random start point.
    startx = random.randint(5, CELLWIDTH - 6)
    starty = random.randint(5, CELLHEIGHT - 6)
    wormCoords = [{'x': startx,     'y': starty},
                  {'x': startx - 1, 'y': starty},
                  {'x': startx - 2, 'y': starty}]
    direction = RIGHT

    # Start the apple in a random place.
    apple = getRandomLocation()

    while True: # main game loop
        for event in pygame.event.get(): # event handling loop
            if event.type == QUIT:
                terminate()
            elif event.type == KEYDOWN:
                if (event.key == K_LEFT or event.key == K_a) and direction != RIGHT:
                    direction = LEFT
                elif (event.key == K_RIGHT or event.key == K_d) and direction != LEFT:
                    direction = RIGHT
                elif (event.key == K_UP or event.key == K_w) and direction != DOWN:
                    direction = UP
                elif (event.key == K_DOWN or event.key == K_s) and direction != UP:
                    direction = DOWN
                elif event.key == K_ESCAPE:
                    terminate()

        # check if the worm has hit itself or the edge
        if wormCoords[HEAD]['x'] == -1 or wormCoords[HEAD]['x'] == CELLWIDTH or wormCoords[HEAD]['y'] == -1 or wormCoords[HEAD]['y'] == CELLHEIGHT:
            return # game over
        for wormBody in wormCoords[1:]:
            if wormBody['x'] == wormCoords[HEAD]['x'] and wormBody['y'] == wormCoords[HEAD]['y']:
                return # game over

        # check if worm has eaten an apply
        if wormCoords[HEAD]['x'] == apple['x'] and wormCoords[HEAD]['y'] == apple['y']:
            # don't remove worm's tail segment
            apple = getRandomLocation() # set a new apple somewhere
        else:
            del wormCoords[-1] # remove worm's tail segment

        # move the worm by adding a segment in the direction it is moving
        if direction == UP:
            newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] - 1}
        elif direction == DOWN:
            newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] + 1}
        elif direction == LEFT:
            newHead = {'x': wormCoords[HEAD]['x'] - 1, 'y': wormCoords[HEAD]['y']}
        elif direction == RIGHT:
            newHead = {'x': wormCoords[HEAD]['x'] + 1, 'y': wormCoords[HEAD]['y']}
        wormCoords.insert(0, newHead)
        DISPLAYSURF.fill(BGCOLOR)
        drawGrid()
        drawWorm(wormCoords)
        drawApple(apple)
        drawScore(len(wormCoords) - 3)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
```

PyGame Key Constants
<https://www.pygame.org/docs/ref/key.html#key-constants-label>

Constant      |ASCII   |Description
--------------|--------|-----------
K_BACKSPACE   |\b      |backspace
K_TAB         |\t      |tab
K_CLEAR       |        |clear
K_RETURN      |\r      |return
K_PAUSE       |        |pause
K_ESCAPE      |^[      |escape
K_SPACE       |        |space
K_EXCLAIM     |!       |exclaim
K_QUOTEDBL    |"       |quotedbl
K_HASH        |#       |hash
K_DOLLAR      |$       |dollar
K_AMPERSAND   |&       |ampersand
K_QUOTE       |        |quote
K_LEFTPAREN   |(       |left parenthesis
K_RIGHTPAREN  |)       |right parenthesis
K_ASTERISK    |*       |asterisk
K_PLUS        |+       |plus sign
K_COMMA       |,       |comma
K_MINUS       |-       |minus sign
K_PERIOD      |.       |period
K_SLASH       |/       |forward slash
K_0           |0       |0
K_1           |1       |1
K_2           |2       |2
K_3           |3       |3
K_4           |4       |4
K_5           |5       |5
K_6           |6       |6
K_7           |7       |7
K_8           |8       |8
K_9           |9       |9
K_COLON       |:       |colon
K_SEMICOLON   |;       |semicolon
K_LESS        |<       |less-than sign
K_EQUALS      |=       |equals sign
K_GREATER     |>       |greater-than sign
K_QUESTION    |?       |question mark
K_AT          |@       |at
K_LEFTBRACKET |[       |left bracket
K_BACKSLASH   |\       |backslash
K_RIGHTBRACKET| ]      |right bracket
K_CARET       |^       |caret
K_UNDERSCORE  |_       |underscore
K_BACKQUOTE   |`       |grave
K_a           |a       |a
K_b           |b       |b
K_c           |c       |c
K_d           |d       |d
K_e           |e       |e
K_f           |f       |f
K_g           |g       |g
K_h           |h       |h
K_i           |i       |i
K_j           |j       |j
K_k           |k       |k
K_l           |l       |l
K_m           |m       |m
K_n           |n       |n
K_o           |o       |o
K_p           |p       |p
K_q           |q       |q
K_r           |r       |r
K_s           |s       |s
K_t           |t       |t
K_u           |u       |u
K_v           |v       |v
K_w           |w       |w
K_x           |x       |x
K_y           |y       |y
K_z           |z       |z
K_DELETE      |        |delete
K_KP0         |        |keypad 0
K_KP1         |        |keypad 1
K_KP2         |        |keypad 2
K_KP3         |        |keypad 3
K_KP4         |        |keypad 4
K_KP5         |        |keypad 5
K_KP6         |        |keypad 6
K_KP7         |        |keypad 7
K_KP8         |        |keypad 8
K_KP9         |        |keypad 9
K_KP_PERIOD   |.       |keypad period
K_KP_DIVIDE   |/       |keypad divide
K_KP_MULTIPLY |*       |keypad multiply
K_KP_MINUS    |-       |keypad minus
K_KP_PLUS     |+       |keypad plus
K_KP_ENTER    |\r      |keypad enter
K_KP_EQUALS   |=       |keypad equals
K_UP          |        |up arrow
K_DOWN        |        |down arrow
K_RIGHT       |        |right arrow
K_LEFT        |        |left arrow
K_INSERT      |        |insert
K_HOME        |        |home
K_END         |        |end
K_PAGEUP      |        |page up
K_PAGEDOWN    |        |page down
K_F1          |        |F1
K_F2          |        |F2
K_F3          |        |F3
K_F4          |        |F4
K_F5          |        |F5
K_F6          |        |F6
K_F7          |        |F7
K_F8          |        |F8
K_F9          |        |F9
K_F10         |        |F10
K_F11         |        |F11
K_F12         |        |F12
K_F13         |        |F13
K_F14         |        |F14
K_F15         |        |F15
K_NUMLOCK     |        |numlock
K_CAPSLOCK    |        |capslock
K_SCROLLOCK   |        |scrollock
K_RSHIFT      |        |right shift
K_LSHIFT      |        |left shift
K_RCTRL       |        |right control
K_LCTRL       |        |left control
K_RALT        |        |right alt
K_LALT        |        |left alt
K_RMETA       |        |right meta
K_LMETA       |        |left meta
K_LSUPER      |        |left Windows key
K_RSUPER      |        |right Windows key
K_MODE        |        |mode shift
K_HELP        |        |help
K_PRINT       |        |print screen
K_SYSREQ      |        |sysrq
K_BREAK       |        |break
K_MENU        |        |menu
K_POWER       |        |power
K_EURO        |        |Euro
K_AC_BACK     |        |Android back button

## Tetromino

``` python
# Board is a 2D array with char '.' as blank and 'o' as block
BOARDWIDTH = 10
BOARDHEIGHT = 20
BLANK = '.'
# Each brick is defined in templates (3D char array for rotation)
S_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '..OO.',
                     '.OO..',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..OO.',
                     '...O.',
                     '.....']]
# Each piece is an object with shape, rotation, x, y, and color
def getNewPiece():
    # return a random new piece in a random rotation and color
    shape = random.choice(list(PIECES.keys()))
    newPiece = {'shape': shape,
                'rotation': random.randint(0, len(PIECES[shape]) - 1),
                'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                'y': -2, # start it above the board (i.e. less than 0)
                'color': random.randint(0, len(COLORS)-1)}
    return newPiece

# colliding detection for each block in piece
def isValidPosition(board, piece, adjX=0, adjY=0):
    # Return True if the piece is within the board and not colliding
    for x in range(TEMPLATEWIDTH):
        for y in range(TEMPLATEHEIGHT):
            isAboveBoard = y + piece['y'] + adjY < 0
            if isAboveBoard or PIECES[piece['shape']][piece['rotation']][y][x] == BLANK:
                continue
            if not isOnBoard(x + piece['x'] + adjX, y + piece['y'] + adjY):
                return False
            if board[x + piece['x'] + adjX][y + piece['y'] + adjY] != BLANK:
                return False
    return True

def isCompleteLine(board, y):
    # Return True if the line filled with boxes with no gaps.
    for x in range(BOARDWIDTH):
        if board[x][y] == BLANK:
            return False
    return True

def runGame():
    # setup variables for the start of the game
    board = getBlankBoard()
    fallingPiece = getNewPiece()
    nextPiece = getNewPiece()
    ...

    while True: # game loop
        if fallingPiece == None:
            # No falling piece in play, so start a new piece at the top
            fallingPiece = nextPiece
            nextPiece = getNewPiece()
            lastFallTime = time.time() # reset lastFallTime

            if not isValidPosition(board, fallingPiece):
                return # can't fit a new piece on the board, so game over

        for event in pygame.event.get(): # event handling loop
            ...
            elif event.type == KEYDOWN:
                # moving the piece sideways
                if (event.key == K_LEFT or event.key == K_a) and isValidPosition(board, fallingPiece, adjX=-1):
                    fallingPiece['x'] -= 1
                    movingLeft = True
                    movingRight = False
                    lastMoveSidewaysTime = time.time()
                elif (event.key == K_RIGHT or event.key == K_d) and isValidPosition(board, fallingPiece, adjX=1):
                    fallingPiece['x'] += 1
                    movingRight = True
                    movingLeft = False
                    lastMoveSidewaysTime = time.time()
                # rotating the piece (if there is room to rotate)
                elif (event.key == K_UP or event.key == K_w):
                    fallingPiece['rotation'] = (fallingPiece['rotation'] + 1) % len(PIECES[fallingPiece['shape']])
                    if not isValidPosition(board, fallingPiece):
                        fallingPiece['rotation'] = (fallingPiece['rotation'] - 1) % len(PIECES[fallingPiece['shape']])

        # handle moving the piece because of user input
        if (movingLeft or movingRight) and time.time() - lastMoveSidewaysTime > MOVESIDEWAYSFREQ:
            if movingLeft and isValidPosition(board, fallingPiece, adjX=-1):
                fallingPiece['x'] -= 1
            elif movingRight and isValidPosition(board, fallingPiece, adjX=1):
                fallingPiece['x'] += 1
            lastMoveSidewaysTime = time.time()

        # let the piece fall if it is time to fall
        if time.time() - lastFallTime > fallFreq:
            # see if the piece has landed
            if not isValidPosition(board, fallingPiece, adjY=1):
                # falling piece has landed, set it on the board
                addToBoard(board, fallingPiece)
                score += removeCompleteLines(board)
                level, fallFreq = calculateLevelAndFallFreq(score)
                fallingPiece = None
            else:
                # piece did not land, just move the piece down
                fallingPiece['y'] += 1
                lastFallTime = time.time()

        # drawing everything on the screen
        drawBoard(board)
        drawStatus(score, level)
        drawNextPiece(nextPiece)
        if fallingPiece != None:
            drawPiece(fallingPiece)

        pygame.display.update()
        FPSCLOCK.tick(FPS)
```

## Squirrel Eat Squirrel

Nothing new, changed to germ eat germ.

## Star Pusher

<!--
TODO:
unfinished
-->

## Four Extra Games
