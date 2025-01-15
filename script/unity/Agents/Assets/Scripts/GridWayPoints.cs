using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GridWayPoints : MonoBehaviour
{
    public Node[,] grid;
    List<Node> path = new List<Node>();
    int curNode = 0;

    public GameObject prefabWaypoint;
    public Material goalMat;
    public Material wallMat;

    Vector3 goal;
    float speed = 4.0f;
    float accuracy = 0.5f;
    float rotSpeed = 4f;

    int spacing = 5;

    Node startNode;
    Node endNode;

    void Start()
    {
        // create grid
        grid = new Node[,] {
            { new Node(), new Node(), new Node(false), new Node(), new Node(), new Node() },
            { new Node(), new Node(false), new Node(), new Node(), new Node(), new Node() },
            { new Node(), new Node(false), new Node(), new Node(), new Node(), new Node() },
            { new Node(), new Node(), new Node(), new Node(false), new Node(), new Node() },
            { new Node(), new Node(), new Node(), new Node(), new Node(false), new Node() },
            { new Node(), new Node(), new Node(false), new Node(), new Node(false), new Node() },
            { new Node(), new Node(false), new Node(false), new Node(), new Node(), new Node() }
        };

        // initialize grid points
        for (int i = 0; i < grid.GetLength(0); i++)
        {
            for (int j = 0; j < grid.GetLength(1); j++)
            {
                grid[i, j].Waypoint = Instantiate(
                    prefabWaypoint,
                    new Vector3(
                        i * spacing,
                        0.1f, //this.transform.position.y,
                        j * spacing),
                    Quaternion.identity);

                if (!grid[i, j].Walkable)
                {
                    grid[i, j].Waypoint.GetComponent<Renderer>().material = wallMat;
                }
                else
                {
                    grid[i, j].Neighbors = Node.GetAdjacentNodes(grid, i, j);
                }
            }
        }

        startNode = grid[0, 0];
        endNode = grid[6, 5];
        startNode.Walkable = true;
        endNode.Walkable = true;
        endNode.Waypoint.GetComponent<Renderer>().material = goalMat;

        this.transform.position = new Vector3(
            startNode.Waypoint.transform.position.x,
            this.transform.position.y,
            startNode.Waypoint.transform.position.z);
    }

    void LateUpdate()
    {
        // calculate the shortest path when the return key is pressed
        if (Input.GetKeyDown(KeyCode.Return))
        {
            this.transform.position = new Vector3(
                startNode.Waypoint.transform.position.x,
                this.transform.position.y,
                startNode.Waypoint.transform.position.z);
            curNode = 0;
            path = Node.BFS(startNode, endNode);
        }

        // if there's no path, do nothing
        if (path.Count == 0) return;

        // set the goal position
        goal = new Vector3(
            path[curNode].Waypoint.transform.position.x,
            this.transform.position.y,
            path[curNode].Waypoint.transform.position.z);

        // set the direction
        Vector3 direction = goal - this.transform.position;

        // move toward the goal or increase the counter to set another goal in the next iteration
        if (direction.magnitude > accuracy)
        {
            this.transform.rotation = Quaternion.Slerp(
                this.transform.rotation,
                Quaternion.LookRotation(direction),
                Time.deltaTime * rotSpeed);
            this.transform.Translate(0, 0, speed * Time.deltaTime);
        }
        else
        {
            if (curNode < path.Count - 1)
            {
                curNode++;
            }
        }
    }

    /*
    void LateUpdate()
    {
        // calculate the shortest path when the return key is pressed
        if (Input.GetKeyDown(KeyCode.Return))
        {
            this.transform.position = new Vector3(
                startNode.Waypoint.transform.position.x,
                this.transform.position.y,
                startNode.Waypoint.transform.position.z);
            curNode = 0;
            path.Add(grid[0, 1]);
            path.Add(endNode);
        }

        // if there's no path, do nothing
        if (path.Count == 0) return;

        // set the goal position
        goal = new Vector3(
            path[curNode].Waypoint.transform.position.x,
            this.transform.position.y,
            path[curNode].Waypoint.transform.position.z);

        // set the direction
        Vector3 direction = goal - this.transform.position;

        // move toward the goal or increase the counter to set another goal in the next iteration
        if (direction.magnitude > accuracy)
        {
            this.transform.rotation = Quaternion.Slerp(
                this.transform.rotation,
                Quaternion.LookRotation(direction),
                Time.deltaTime * rotSpeed);
            this.transform.Translate(0, 0, speed * Time.deltaTime);
        }
        else
        {
            if (curNode < path.Count - 1)
            {
                curNode++;
            }
        }
    }
    */
}
