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

## Flocks and Crowds dynamics

It is better to represent the entire crowd as an entity rather than trying to model each individual as its own agent. Each individual in the group only really needs to know where the group is heading and what their nearest neighbor is up to in order to function as part of the system.

## Behavior Trees

Behavior trees are a collection of nodes organized in a hierarchical order, in which nodes are connected to parents rather than states connected to each other, resembling branches on a tree, hence the name.

- Selector task
- Sequence tasks
- Parallel tasks
- Decorators

## Fuzzy Logic
