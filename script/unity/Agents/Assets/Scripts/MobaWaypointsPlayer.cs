using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MobaWaypointsPlayer : MonoBehaviour
{
    public float speed;
    private List<GameObject> wayPointsList;
    private Transform target;
    private GameObject[] wayPoints;

    void Start()
    {
        target = GameObject.FindGameObjectWithTag("target").transform;
        wayPointsList = new List<GameObject>();
        wayPoints = GameObject.FindGameObjectsWithTag("wayPoint");
        foreach (GameObject newWayPoint in wayPoints)
        {
            wayPointsList.Add(newWayPoint);
        }
    }

    void Update()
    {
        Follow();
    }

    void Follow()
    {
        GameObject wayPoint = null;
        if (Physics.Linecast(transform.position, target.position))
        {
            wayPoint = findBestPath();
        }
        else
        {
            wayPoint = GameObject.FindGameObjectWithTag("target");
        }
        Vector3 Dir = (wayPoint.transform.position -
        transform.position).normalized;
        transform.position += Dir * Time.deltaTime * speed;
        transform.rotation = Quaternion.LookRotation(Dir);
    }

    GameObject findBestPath()
    {
        GameObject bestPath = null;
        float distanceToBestPath = Mathf.Infinity;
        foreach (GameObject go in wayPointsList)
        {
            float distToWayPoint = Vector3.
                Distance(transform.position, go.transform.position);
            float distWayPointToTarget = Vector3.
                Distance(go.transform.position, target.transform.position);
            float distToTarget = Vector3.
                Distance(transform.position, target.position);
            bool wallBetween = Physics.Linecast(transform.position, go.transform.position);
            if ((distToWayPoint < distanceToBestPath)
                && (distToTarget > distWayPointToTarget)
                && (!wallBetween))
            {
                distanceToBestPath = distToWayPoint;
                bestPath = go;
            }
            else
            {
                bool wayPointToTargetCollision = Physics.Linecast(go.transform.position, target.position);
                if (!wayPointToTargetCollision)
                {
                    bestPath = go;
                }
            }
        }
        return bestPath;
    }
}
