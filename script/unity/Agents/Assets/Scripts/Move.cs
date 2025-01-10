using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move : MonoBehaviour
{
    Vector3 goal;
    float speed = 1.0f;
    float accuracy = 0.1f;
    float rotSpeed = 2f;

    // Start is called before the first frame update
    void Start()
    {
        goal = this.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        RaycastHit hit;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

        if (Physics.Raycast(ray, out hit) && Input.GetMouseButtonDown(0))
        {
            //Vector3 newPosition = new Vector3(hit.point.x, this.transform.position.y, hit.point.z);
            //this.transform.position = newPosition;
            // set goal to be the point on plane that mouse clicked
            goal = new Vector3(hit.point.x, this.transform.position.y, hit.point.z);
            Debug.Log("New position vector: " + goal.ToString());
        }

        //this.transform.LookAt(goal);
        Vector3 direction = goal - this.transform.position;

        if (Vector3.Distance(transform.position, goal) > accuracy)
        {
            var goalRotation = Quaternion.LookRotation(direction);
            this.transform.rotation = Quaternion.Slerp(this.transform.rotation, goalRotation, Time.deltaTime * rotSpeed);
            this.transform.Translate(0, 0, speed * Time.deltaTime);
        }
    }
}
