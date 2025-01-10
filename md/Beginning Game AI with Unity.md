# Beginning Game AI with Unity, Programming Artificial Intelligence with CSharp

Source Code: <https://github.com/Apress/beginning-game-ai-unity>

## Basics

### Intelligent Agent

- Processing perceived information from environment
- Taking action in order to reach a specific goal

`P* -> A`

- `P*`: percept sequence
- `A`: appropriate action

### Finite State Machine (FSM) for NPCs

- Easy to implement
- Low impact on performances
- Genuine and credible user experience when will designed

### Artificial Life (AL) Game: 'Creatures' series by Steve Grand

- genetics algorithms
- neural networks
- create virtual-biological beings
- transmit their genetic traits to offspring by mating
- learn by experience

### Move and Turn

movement vector like this:

`myObject.transform.Translate(0, 0, speed * Time.deltaTime);`

and turn the object to face the target position:

`this.transform.LookAt(positionToLookAt);`

There are two very popular techniques to implement linear interpolation to rotate an object:

- `Lerp` (Linear intERPolation)
- `Slerp` (Spherical Linear intERPolation)

For example, use slerp to calculate rotation:

`Quaternion.Slerp(startingRotation, goalRotation, rotationSpeed);`

## Paths and Waypoints

## Navigation

## Behaviors
