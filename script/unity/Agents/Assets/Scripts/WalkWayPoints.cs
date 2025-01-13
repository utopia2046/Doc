using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
