# Unity 2017 Game AI Programming (Third Edition) Reading Notes

[TOC]

- [Unity 2017 Game AI Programming (Third Edition) Reading Notes](#unity-2017-game-ai-programming-third-edition-reading-notes)
  - [Finite State Machines](#finite-state-machines)
  - [Implementing Sensors](#implementing-sensors)
  - [Path Finding](#path-finding)
    - [Use Layer for Obstacles Avoidance](#use-layer-for-obstacles-avoidance)
    - [NavMesh](#navmesh)
  - [Flocks and Crowds dynamics](#flocks-and-crowds-dynamics)
    - [Reynolds algorithm](#reynolds-algorithm)
  - [Behavior Trees](#behavior-trees)
  - [Fuzzy Logic](#fuzzy-logic)
    - [Membership Function](#membership-function)
    - [Fuzzy logic controller](#fuzzy-logic-controller)

Source Code: <https://github.com/PacktPublishing/Unity-2017-Game-AI-Programming-Third-Edition>

## Finite State Machines

- States: This component defines a set of distinct states that a game entity or an NPC can choose from (patrol, chase, and shoot)
- Transitions: This component defines relations between different states
- Rules: This component is used to trigger a state transition (player on sight, close enough to attack, and lost/killed player)
- Events: This is the component that will trigger to check the rules (guard's visible area, distance to the player, and so on)

In Unity, state machine could be implemented using `Animator`, although there is no animation needed.

For each state in Animator state machine, a script inherited from `StateMachineBehaviour` could be attached. This script could contain override event handlers like:

- `OnStateEnter`
- `OnStateUpdate`
- `OnStateExit`

``` csharp
// Tower object has an Animator with a boolean parameter TankInRange to determine if the player is in tower's shoot range,
// and state LockOn has below script attached
public class LockedOnState : StateMachineBehaviour
{
    GameObject player;
    Tower tower;

    // OnStateEnter is called when a transition starts and the state machine starts to evaluate this state
    override public void OnStateEnter(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        player = GameObject.FindWithTag("Player");
        tower = animator.gameObject.GetComponent<Tower>();
        tower.LockedOn = true;
    }

    //OnStateUpdate is called on each Update frame between OnStateEnter and OnStateExit callbacks
    override public void OnStateUpdate(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        animator.gameObject.transform.LookAt(player.transform);
    }

    // OnStateExit is called when a transition ends and the state machine finishes evaluating this state
    override public void OnStateExit(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        animator.gameObject.transform.rotation = Quaternion.identity;
        tower.LockedOn = false;
    }
}

// Also, in Tower's MonoScript, the collider event will set the animator parameter and trigger state change
public class Tower : MonoBehaviour {
    ...
    private void OnTriggerEnter(Collider other) {
        if (other.tag == "Player") {
            animator.SetBool("TankInRange", true);
        }
    }

    private void OnTriggerExit(Collider other) {
        if (other.tag == "Player") {
            animator.SetBool("TankInRange", false);
        }
    }
    ...
}
```

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

### Reynolds algorithm

Three basic concepts that define how a flock works

- Separation: This means maintaining a distance with other neighbors in the flock to avoid collision.
- Alignment: This means to moving in the same direction as the flock, and with the same velocity.
- Cohesion: This means maintaining a maximum distance from the flock's center.

## Behavior Trees

Behavior trees are a collection of nodes organized in a hierarchical order, in which nodes are connected to parents rather than states connected to each other, resembling branches on a tree, hence the name.

- Selector task
- Sequence tasks
- Parallel tasks
- Decorators

A node will always return one of the following states:

1. Success
2. Failure
3. Running, validity not determined, in waiting status (until asynchronous checking finishes)

Composite nodes: node with one or more children.

- Sequence (AND): any of the children at any step of the sequence return false, the sequence itself will report a failure.
- Selectors (OR): If any one of the children nodes returns true, the selector returns true immediately, without evaluating any more children.

Decorator node: exactly one child only, takes the state returned by the child and evaluates the response based on its own parameters.

- Inverter (NOT): takes the opposite of the state returned by its child.
- Repeater (while): repeats the evaluation of the child a specified (or infinite) number of times until it evaluates as either TRUE or FALSE.
- Limiter: limits the number of times a node will be evaluated to avoid getting an agent stuck in an awkward infinite behavior loop.
- (debugging and testing) Fake state: always evaluates true or false.
- (debugging and testing) Breakpoint.

Behavior Trees in Unity Asset Store

- Behave by Angry Ant
- Behavior Machine
- Behavior Designer

## Fuzzy Logic

### Membership Function

It allow us to determine how true a statement is, using logical chunks of information raw values.

Example: determine the **degree of membership** to a set. Three states are evaluated to determine how true each one is, and which is the most true

- Player is in a critical condition
- Player is hurt
- Player is health

| Tips: each statement evaluation function could be specified as an `AnimationCurve` and set in Editor. More info: <http://docs.unity3d.com/ScriptReference/AnimationCurve.html>

``` csharp
// statement evaluatation code
public void EvaluateStatements()
{
    if (string.IsNullOrEmpty(healthInput.text))
    {
        return;
    }
    float inputValue = float.Parse(healthInput.text);

    healthyValue = healthy.Evaluate(inputValue);
    hurtValue = hurt.Evaluate(inputValue);
    criticalValue = critical.Evaluate(inputValue);

    SetLabels();
}
```

You could predefine these rules or have some degree of randomness determine the limits, and every single agent would behave uniquely and respond to things in their own way.

We could also build a behavior tree using fuzzy logic to evaluate each node. We end up with a very flexible, powerful, and nuanced AI system by combining these two concepts.

### Fuzzy logic controller

| Tips: In MonoScript, each public property will be editable in Inspector. Also, there are more options:

1. Add `Header` attribute to group properties in Inspector;
2. Use `SerializeField` attribute for complex types properties;

``` csharp
[Header("Morality Gradient")]
[SerializeField]
private AnimationCurve good;
[SerializeField]
private AnimationCurve neutral;
[SerializeField]
private AnimationCurve evil;
```
