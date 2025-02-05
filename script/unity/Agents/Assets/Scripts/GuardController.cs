using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GuardController : MonoBehaviour
{
    public Transform player;
    float fovDist = 20.0f;
    float fovAngle = 45.0f;

    // Start is called before the first frame update
    void Start()
    {

    }

    void Update()
    {
        if (ICanSee(player))
        {
            Debug.Log("I saw the player at " + player.position);
        }
        else
        {
            Debug.Log("All quiet here...");
        }
    }

    bool ICanSee(Transform player)
    {
        Vector3 direction = player.position - this.transform.position;
        float angle = Vector3.Angle(direction, this.transform.forward);

        RaycastHit hit;

        if (Physics.Raycast(this.transform.position, direction, out hit) && // Can I cast a ray from my position to the player's position?
            hit.collider.gameObject.tag == "Player" && // Did the ray hit the player?
            direction.magnitude < fovDist && // Is the player close enough to be seen?
            angle < fovAngle // Is the player in the view cone?
        )
        {
            return true;
        }

        return false;
    }

}
