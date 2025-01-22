using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class AgentController : MonoBehaviour
{
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            RaycastHit hit;
            if (Physics.Raycast(
                Camera.main.ScreenPointToRay(Input.mousePosition),
                out hit,
                100))
            {
                // tell NavMesh to calculate the best path to the mouse clicked position
                this.GetComponent<NavMeshAgent>().SetDestination(hit.point);
            }
        }
    }
}
