using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class AgentController : MonoBehaviour
{
    public float knockRadius = 20.0f;

    void Update()
    {
        // when user click on map, navigate player to the point
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

        // when user press space, hit
        if (Input.GetKey("space"))
        {
            StartCoroutine(PlayKnock()); // Play audio file

            // Create the sphere collider
            Collider[] hitColliders = Physics.OverlapSphere(transform.position, knockRadius);
            for (int i = 0; i < hitColliders.Length; i++) // check the collisions
            {
                // If it's a guard, trigger the Investigation!
                if (hitColliders[i].tag == "guard")
                {
                    hitColliders[i].GetComponent<GuardController>().InvestigatePoint(this.transform.position);
                }
            }
        }
    }

    IEnumerator PlayKnock()
    {
        AudioSource audio = GetComponent<AudioSource>();

        audio.Play();
        yield return new WaitForSeconds(audio.clip.length);
    }
}
