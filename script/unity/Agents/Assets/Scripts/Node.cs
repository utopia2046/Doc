using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Node
{
    private int depth;
    private bool walkable;

    private GameObject waypoint = new GameObject();
    private List<Node> neighbors = new List<Node>();

    public int Depth { get => depth; set => depth = value; }
    public bool Walkable { get => walkable; set => walkable = value; }

    public GameObject Waypoint { get => waypoint; set => waypoint = value; }
    public List<Node> Neighbors { get => neighbors; set => neighbors = value; }

    public Node()
    {
        this.depth = -1;
        this.walkable = true;
    }

    public Node(bool walkable)
    {
        this.depth = -1;
        this.walkable = walkable;
    }

    public override bool Equals(System.Object obj)
    {
        if (obj == null) return false;

        Node n = obj as Node;
        if ((System.Object)n == null)
        {
            return false;
        }
        if (this.waypoint.transform.position.x == n.Waypoint.transform.position.x &&
            this.waypoint.transform.position.z == n.Waypoint.transform.position.z)
        {
            return true;
        }

        return false;
    }

    public static List<Node> GetAdjacentNodes(Node[,] m, int i, int j)
    {
        List<Node> l = new List<Node>();

        // node up
        if ((i - 1 >= 0) && (m[i - 1, j].Walkable))
        {
            l.Add(m[i - 1, j]);
        }

        // node down
        if ((i + 1 < m.GetLength(0)) && (m[i + 1, j].Walkable))
        {
            l.Add(m[i + 1, j]);
        }

        // node left
        if ((j - 1 >= 0) && (m[i, j - 1].Walkable))
        {
            l.Add(m[i, j - 1]);
        }

        // node right
        if ((j + 1 < m.GetLength(1)) && (m[i, j + 1].Walkable))
        {
            l.Add(m[i, j + 1]);
        }

        return l;
    }

    public static List<Node> BFS(Node start, Node end)
    {
        Queue<Node> toVisit = new Queue<Node>();
        List<Node> visited = new List<Node>();

        Node currentNode = start;
        currentNode.Depth = 0;
        toVisit.Enqueue(currentNode);

        List<Node> finalPath = new List<Node>();

        while (toVisit.Count > 0)
        {
            currentNode = toVisit.Dequeue();

            if (visited.Contains(currentNode))
                continue;

            visited.Add(currentNode);

            if (currentNode.Equals(end))
            {
                while (currentNode.Depth != 0)
                {
                    foreach (Node n in currentNode.Neighbors)
                    {
                        if (n.Depth == currentNode.Depth - 1)
                        {
                            finalPath.Add(currentNode);
                            currentNode = n;
                            break;
                        }
                    }
                }
                finalPath.Reverse();
                break;
            }
            else
            {
                foreach (Node n in currentNode.Neighbors)
                {
                    if (!visited.Contains(n) && n.Walkable)
                    {
                        n.Depth = currentNode.Depth + 1;
                        toVisit.Enqueue(n);
                    }
                }
            }
        }
        return finalPath;
    }

}
