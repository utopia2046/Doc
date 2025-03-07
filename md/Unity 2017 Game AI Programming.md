# Unity 2017 Game AI Programming (Third Edition) Reading Notes

Source Code: <https://github.com/PacktPublishing/Unity-2017-Game-AI-Programming-Third-Edition>

## Finite State Machines

- States: This component defines a set of distinct states that a game entity or an NPC can choose from (patrol, chase, and shoot)
- Transitions: This component defines relations between different states
- Rules: This component is used to trigger a state transition (player on sight, close enough to attack, and lost/killed player)
- Events: This is the component that will trigger to check the rules (guard's visible area, distance to the player, and so on)

## Implementing Sensors

## Path Finding

- Dijkstra Shortest Path: Visualization: <https://www.cs.usfca.edu/~galles/visualization/Dijkstra.html>
- A* (A star)
- Unity: Navigation Mesh generation and the NavMesh agent
- IDA*: iterative deepening A* (depth first, less memory cost than A* and slower)

### Use Layer for Obstacles Avoidance

Layers could be used by Raycast to determine which GameObjects are obstacles and which are not. For example, set obstacles on Layer 8 and name the layer "Obstacles". Then in code, we could detect if an obstacle is in path like below:

``` cs
private void ApplyAvoidance(ref Vector3 direction)
{
    //Only detect layer 8 (Obstacles)
    //We use bitshifting to create a layermask with a value of
    //0100000000 where only the 8th position is 1, so only it is active.
    int layerMask = 1 << 8;

    //Check that the agent hit with the obstacles within it's minimum distance to avoid
    if (Physics.Raycast(transform.position, transform.forward, out avoidanceHit, minimumAvoidanceDistance, layerMask))
    {
        //Get the normal of the hit point to calculate the new direction
        hitNormal = avoidanceHit.normal;
        hitNormal.y = 0.0f; //Don't want to move in Y-Space

        //Get the new direction vector by adding force to agent's current forward vector
        direction = transform.forward + hitNormal * force;
    }
}
```

### NavMesh

1. Set all floors, walls, obstacles etc. to be `Navigation Static` (top right on Inspector);
2. Open `Window -> AI -> Navigation` panel to bake NavMesh;
3. Add `NavMeshAgent` component on player or NPC that need to move to certain target position using NavMesh;
4. Add `MonoBehavior` script that respond to user input like mouse click;

``` cs
private NavMeshAgent[] navAgents;
public Transform targetMarker; // select target object in Editor

private void Start()
{
    // find all GameObjects with NavMeshAgent on it
    navAgents = FindObjectsOfType(typeof(NavMeshAgent)) as NavMeshAgent[];
}

private void Update()
{
    if(Input.GetMouseButtonDown(0))
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hitInfo;

        if (Physics.Raycast(ray.origin, ray.direction, out hitInfo))
        {
            Vector3 targetPosition = hitInfo.point;
            UpdateTargets(targetPosition);
            targetMarker.position = targetPosition;
        }
    }
}

// set destination for all NavMeshAgents
private void UpdateTargets(Vector3 targetPosition)
{
    foreach(NavMeshAgent agent in navAgents)
    {
        agent.destination = targetPosition;
    }
}
```

Add OffMeshLink

1. Add GameObject for jump start point and end point;
2. On one of the platform, add `OffMeshLink` component, select `Start` and `End`;
3. Set `Cost Override` to be positive and `Activated` as true, now the agent will be able to jump from start to end;
4. If necessary, set `Bidirectional`;

## Flocks and Crowds dynamics

It is better to represent the entire crowd as an entity rather than trying to model each individual as its own agent. Each individual in the group only really needs to know where the group is heading and what their nearest neighbor is up to in order to function as part of the system.

## Behavior Trees

Behavior trees are a collection of nodes organized in a hierarchical order, in which nodes are connected to parents rather than states connected to each other, resembling branches on a tree, hence the name.

- Selector task
- Sequence tasks
- Parallel tasks
- Decorators

## Fuzzy Logic
