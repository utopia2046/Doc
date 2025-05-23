# Practical Game AI Programming

## Summary

- Chapter 1, Different Problems Require Different Solutions, is a brief introduction to the video game industry and game AI.
- Chapter 2, Possibility and Probability Maps, focuses on how to create and use probability and possibility maps for AI characters.
- Chapter 3, Production Systems, describes how to create a set of rules necessary for the character AI to achieve their goals.
- Chapter 4, Environment and AI, focuses on the interaction between the characters in the game and their environment.
- Chapter 5, Animation Behaviors, shows best practices to implement animations in our game.
- Chapter 6, Navigation Behavior and Pathfinding, focuses on how to calculate the best options for the AI to move in real time.
- Chapter 7, Advanced Pathfinding, focuses on the use of theta algorithms to find short and realistic-looking paths.Preface
- Chapter 8, Crowd Interactions, focuses on how the AI should behave when there are a lot of characters on the same scene.
- Chapter 9, AI Planning and Collision Avoidance, discusses the anticipation of the AI, knowing in advance what they will do when arriving at a position or facing a problem.
- Chapter 10, Awareness, focuses on working with awareness systems to create stealth genre mechanics

[Source Code](https://github.com/PacktPublishing/Practical-Game-AI-Programming)

## Possibility and Probability Maps

Possibility map (FPS): series of actions list accordingly to different situations in the game.

``` csharp
public class Enemy : MonoBehaviour {
    private int Health = 100;
    private bool statePassive; // default state
    private bool stateAggressive;
    private bool stateDefensive;
    private bool trigger; // player enter enemy view scope

    // Use this for initialisation
    void Start () {
        statePassive = true;
    }
    // Update is called once per frame
    void Update () {
        // The AI will remain passive until an interaction with the player occurs
        if (Health == 100 && trigger == false)
        {
            statePassive = true;
            stateAggressive = false;
            stateDefensive = false;
            return;
        }
        // The AI will shift to the aggressive mode if player comes and AI is above 20HP
        if (Health > 20 && trigger == true)
        {
            statePassive = false;
            stateAggressive = true;
            stateDefensive = false;
            return;
        }
        // The AI will shift to the defensive mode if player comes or if the AI is below 20 HP
        if (trigger == true || Health <= 20)
        {
            statePassive = false;
            stateAggressive = false;
            stateDefensive = true;
        }
    }
}
```

``` mermaid
graph TD
    P[PASSIVE<hr>Hold Position] -- trigger & low HP --> D[DEFENSIVE<hr>Find Cover<br>Enter Building]
    P -- trigger & healthy --> A[AGGRESSIVE<hr>Face Player<br>Fire<br>Search Player]
    A -- low HP --> D
    D -- healthy --> A
```

A **probability map** is a more complex and detailed version of a possibility map because itrelies on probabilities in order to change the behavior of the character, rather than a simple on or off trigger.

Time      | Guard | Eat/Drink | Walk
----------|-------|-----------|-----
Morning   | 0.87  | 0.1       | 0.03
Afternoon | 0.48  | 0.32      | 0.2
Night     | 0.35  | 0.4       | 0.25

We can alsoupdate this probability every five minutes or so, in case the player stays still, waiting forour enemy to change position.

``` cs
Void Passive () {
    rndNumber = Random.Range(0, 100);
    If (morningTime == true && rndNumber > 13)
    {
        // We have 87% of chance
        goGuard();
    }
    if (morningTime == true && rndNumber =< 13 && rndNumber < 3)
    {
        // We have 10% of chance
        goDrink();
    }
    if (morningTime == true && rndNumber <= 3)
    {
        // We have 3% of chance
        goWalk();
    }
    if (afternoonTime == true)
    {
        //...
    }
    if (nightTime == true)
    {
        //...
    }
}
```

Another special thing that we can do with probability maps, is giving the AI theopportunity to learn from himself, making him smarter every time the player decides toplay the game. If the player confronted the computer 100 timesand 60% of those times he used a grenade, the AI should have that in mind and reactaccording to that probability.

## Automated finite-state machines (AFSMs)

Character will choose the best optionaccording to the factors that he will be able to calculate (position, player HP, currentweapon, and so on). To plan an AFSMs, we break character actions into two or three maincolumns:

1. In one side of the column we put the **main information**, such as orientation, speed, or goals;
2. In the other columns we put **actions that can be performed over the firstcolumn actions**, such as moving, shooting,charging, finding cover, hiding, using the object, and so on.

Doing that, we can ensure that our character can react according to our firstcolumn independently of the position in which he is currently placed.

1. Column 1: Main Targets
   - HP > 20 Defeat Player
   - HP <= 20 Survive
2. Column 2: Secondary Targets
    - Find Player
    - Find Cover
    - Find Points
3. Column 3: Actions
    - Move To
    - Fire
    - Use Object
    - Crouch

Link every action in the third column to the second, and all of the behaviors in the second to the first column goals. Then calculate chances. An example logic will be

``` cs
void Update ()
{
    chanceFire = ((hitBullets / firedBullets) * 100) = 0;
    chanceHit = ((pHitBullets / pFiredBullets) * 100) = 0;
    if (currentHealth > 20 && currentBullets > 5)
    {
        Fire();
    }
    if (currentHealth > 20 && currentBullets < 5 && chanceFire < 80)
    {
        MoveToPoint();
    }
    if (currentHealth > 20 && currentBullets < 5 && chanceFire > 80)
    {
        Fire();
    }
    if (currentHealth > 20 && currentBullets > 5 && chanceFire < 30 && chanceHit > 30)
    {
        MoveToCover();
    }
    if (currentHealth < 20 && currentBullets > 0 && chanceFire > 90 && chanceHit < 50)
    {
        Fire();
    }
}
```

## Theta `A*` algorithm

For each node near current node (begin with starting point), calculate:

- G value, distance from the starting point
- H value, distance from the ending point
- F = G + H

Get the node with smallest F value, which means the shortest path, then move current position to this node. Repeat F value calculation until we get to the ending point

When there are obstacles in the path, we'll explore each node with lowest F, they won't all be on shortest path, so the calculation amount is high, that's why `A*` algorithm is CPU consuming.

``` psuedo code
OPEN // the set of nodes to be evaluated
CLOSED // the set of nodes already evaluated

Add the start node to OPEN

loop
    current = node in OPEN with the lowest f_cost
    remove current from OPEN
    add current to CLOSED

    if current is the target node // path has been found
        return

    foreach neighbor of the current node
        if neighbor is not traversable or neighbor is in CLOSED
        skip to the next neighbor

if new path to neighbor is shorter OR neighbor is not in OPEN
    set f_cost of neighbor
    set parent of neighbor to current
    if neighbor is not in OPEN
        add neighbor to OPEN
```

## Crowd Interactions

Congestion maps vs. Direction Maps

DMs use averagecrowd motion over time to encourage agents to move with the crowd. Because of this, many of the oscillations present in the congestion map approach are smoothly resolved.Conversely, this temporal smoothing prevents DMs from quickly and accurately reacting to changes in the environment and crowd behavior. Both congestion maps and DMs apply the aggregate crowd movement information to the path planning process in much the sameway; however, congestion maps handle agents of varying size and shape, while DMstraditionally assume homogeneity.

DMs使用平均时间上的人群移动来鼓励代理与人群一起移动。因此，拥挤地图方法中存在的许多振荡被平滑地解决了。相反，这种时间平滑防止了 DMs 快速准确地对环境和人群行为的变化做出反应。拥挤地图和 DMs 都在很大程度上将人群移动信息应用于路径规划过程；但是，拥挤地图处理不同大小和形状的代理，而 DMs 传统上假设同质性。

Congestion mapsweight movement penalties proportional to the density of the crowd. Without takingdensity into account, DMs display overly pessimistic pathfinding behavior, where agents are encouraged to path around sparse groups of agents blocking the ideal path.

拥挤地图的权重移动惩罚与人群的密度成正比。不考虑密度，DM 表现出过于悲观的路径搜索行为，其中代理被鼓励绕过稀疏的代理组，这些代理组阻塞了理想路径。

## Opponent actions

Strategies to make the NPC smarter:

1. Having two extra Colliders on each character is to help the other characters identify whether they are looking at the back or front side of the character.
2. Adding a stretched trigger collider in front of the character representing the observation range.
3. Add a parameter called `dangerMeter`, that will help determine whether the situation that our AI character is facing has a higher or lower risk to be attacked.
4. Divide the map into different zones and assign a level of risk that each zones has.
5. Using Sphere Collider (or other kind of Collider) as Trigger Zone, if player enter the zone, instantiate an enemy instace (`OnTriggerEnter`). And when player leave the zone, destroy the instace (`OnTriggerExist`);
6. Use Layer Mask option so that our Raycast can identify what is inside of the character field of view, e.g. Obstacles, Targets, etc.
7. Visualize character view scope in Unity Editor Mode;

``` csharp
using UnityEngine;
using System.Collections;
using UnityEditor;

[CustomEditor (typeof (FieldOfView))]
public class FieldOfViewEditor : Editor {
    void OnSceneGUI() {
        FieldOfView fow = (FieldOfView)target;
        Handles.color = color.white;
        Handles.DrawWireArc(fow.transform.position, Vector3.up, Vector3.forward, 360, fow.viewRadius);
        Vector3 viewAngleA = fow.DirFromAngle(-fow.viewAngle/2, false);
        Handles.DrawLine(fow.transform.position, fow.transform.position + viewAngleA * fow.viewRadius);
        Handles.DrawLine(fow.transform.position, fow.transform.position + viewAngleB * fow.viewRadius);
    }
}

// FieldOfView.cs
public class FieldOfView {
    public float viewRadius;
    [Range(0,360)]
    public float viewAngle;
    public LayerMask targetMask;
    public LayerMask obstacleMask;
    public List<Transform> visibleTargets = new List<Transform>();

    void FindVisibleTargets () {
        visibleTargets.Clear ();
        Collider[] targetInViewRadius = Physics.OverlapSphere(transform.position, viewRadius, targetMask);

        for (int i = 0; i < targetsInViewRadius.Length; i++)
        {
            Transform target = targetInViewRadius [i].transform;
            Vector3 dirToTarget = (target.position - transform.position).normalized;
            if (Vector3.Angle (transform.forward, dirToTarget) < viewAngle / 2) {
                float dstToTarget = Vector3.Distance (transform.position, target.position);
                if (!Physics.Raycast(transform.position, dirToTarget, dstToTarget, obstacleMask)) {
                    visibleTargets.Add(target);
                }
            }
        }
    }

    public Vector3 DirFromAngle(float angleInDegrees, bool angleIsGlobal) {
        if (!angleIsGlobal) {
            angleInDegrees += transform.eulerAngles.y;
        }
        return new Vector3(Mathf.Sin(angleInDegrees * Mathf.Deg2Rad), 0, Mathf.Cos(angleInDegrees * Mathf.Deg2Rad));
    }

    // draw fov in game mode using multiple radius lines, adjust `meshResolution` in Editor to adjust lines number
    void DrawFieldOfView() {
        int stepCount = Mathf.RoundToInt(viewAngle * meshResolution);
        float stepAngleSize = viewAngle / stepCount;
        for (int i = 0; i <= stepCount; i++) {
            float angle = transform.eulerAngles.y - viewAngle / 2 + stepAngleSize * i;
            Debug.DrawLine(transform.position, transform.position + DirFromAngle(angle, true) * viewRadius, Color.red);
        }
    }
}
```
