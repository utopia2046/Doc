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

Waypoints are points on a line of travel that mark specific locations. By connecting waypoint to waypoint we could define a path and guide GameObject to walk along the path.

``` csharp
public class WalkWayPoints : MonoBehaviour
{
    public GameObject[] path; // fill path array with waypoint objects in Editor

    private Vector3 goal;
    private float speed = 4.0f;
    private float accuracy = 0.5f;
    private float rotSpeed = 4f;
    private int curNode = 0;

    // Update is called once per frame
    void Update()
    {
        // get current target waypoint
        goal = new Vector3(
            path[curNode].transform.position.x,
            this.transform.position.y,
            path[curNode].transform.position.z);
        Vector3 direction = goal - this.transform.position;

        if (direction.magnitude > accuracy)
        {
            // turn and move agent toward target waypoint
            this.transform.rotation = Quaternion.Slerp(
                this.transform.rotation,
                Quaternion.LookRotation(direction),
                Time.deltaTime * rotSpeed);
            this.transform.Translate(0, 0, speed * Time.deltaTime);
        }
        else // reach target node, update curNode
        {
            if (curNode < path.Length - 1)
            {
                curNode++;
            }
            else // loop path array
            {
                curNode = 0;
            }
        }
    }
}
```

## Navigation

## Behaviors
