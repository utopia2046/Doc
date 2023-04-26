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

<!--
TODO:
unfinished
-->

## Memory Puzzle

## Slide Puzzle

## Simulate

## Wormy

## Tetromino

## Squirrel Eat Squirrel

## Star Pusher

## Four Extra Games
