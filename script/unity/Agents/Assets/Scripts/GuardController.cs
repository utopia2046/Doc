using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GuardController : MonoBehaviour
{
    // Guard Field of View (FOV) Cone Parameter
    float fovDist = 20.0f;
    float fovAngle = 45.0f;

    // Chasing settings
    public float chasingSpeed = 2.0f;
    public float chasingRotSpeed = 2.0f;
    public float chasingAccuracy = 1.0f;

    // Patrol settings
    public float patrolDistance = 10.0f;
    float patrolWait = 5.0f;
    float patrolTimePassed = 0;

    // Player info
    public Transform player;

    // FSM
    public enum State
    {
        Patrol,
        Investigate,
        Chase
    }

    State curState = State.Patrol;

    // Last place the player was seen
    Vector3 lastPlaceSeen;

    void Start()
    {
        patrolTimePassed = patrolWait;
        lastPlaceSeen = this.transform.position;
    }

    void Update()
    {
        State tmpState = curState;

        if (ICanSee(player))
        {
            Debug.Log("I saw the player at " + player.position);
            curState = State.Chase;
            lastPlaceSeen = player.position;
        }
        else
        {
            Debug.Log("All quiet here...");
            if (curState == State.Chase)
            {
                curState = State.Investigate;
            }
        }

        // State Check
        switch (curState)
        {
            case State.Patrol:
                Patrol();
                break;
            case State.Investigate:
                Investigate();
                break;
            case State.Chase:
                Chase(player);
                break;
        }

        if (tmpState != curState)
        {
            Debug.Log("State change to: " + curState);
        }
    }

    public void InvestigatePoint(Vector3 point)
    {
        lastPlaceSeen = point;
        curState = State.Investigate;
    }

    bool ICanSee(Transform player)
    {
        Vector3 direction = player.position - this.transform.position;
        float angle = Vector3.Angle(direction, this.transform.forward);

        RaycastHit hit;

        // Can I cast a ray from my position to the player's position?
        if (Physics.Raycast(this.transform.position, direction, out hit) &&
            hit.collider.gameObject.tag == "Player" && // Did the ray hit the player?
            direction.magnitude < fovDist && // Is the player close enough to be seen?
            angle < fovAngle // Is the player in the view cone?
        )
        {
            return true;
        }

        return false;
    }

    void Patrol()
    {
        patrolTimePassed += Time.deltaTime;

        if (patrolTimePassed > patrolWait)
        {
            patrolTimePassed = 0; // reset the timer
            Vector3 patrollingPoint = lastPlaceSeen;

            // Generate a random point on the X,Z axis at 'patrolDistance' distance
            // from the lastPlaceSeen position
            patrollingPoint += new Vector3(
                Random.Range(-patrolDistance, patrolDistance),
                0.5f,
                Random.Range(-patrolDistance, patrolDistance));

            // Make the generated point a goal for the agent
            this.GetComponent<UnityEngine.AI.NavMeshAgent>().SetDestination(patrollingPoint);
            Debug.DrawLine(this.transform.position, patrollingPoint, Color.green);
        }
    }

    void Investigate()
    {
        // If the agent arrived at the investigating goal, they should start patrolling there
        if (transform.position == lastPlaceSeen)
        {
            curState = State.Patrol;
        }
        else
        {
            this.GetComponent<UnityEngine.AI.NavMeshAgent>().SetDestination(lastPlaceSeen);
            Debug.Log("Guard's state: " + curState + " point" + lastPlaceSeen);
            Debug.DrawLine(this.transform.position, lastPlaceSeen, Color.green);
        }
    }

    void Chase(Transform player)
    {
        this.GetComponent<UnityEngine.AI.NavMeshAgent>().isStopped = true;
        //this.GetComponent<UnityEngine.AI.NavMeshAgent>().Stop();
        this.GetComponent<UnityEngine.AI.NavMeshAgent>().ResetPath();

        Vector3 direction = player.position - this.transform.position;
        Debug.DrawLine(this.transform.position, player.position, Color.green);

        this.transform.rotation = Quaternion.Slerp(
            this.transform.rotation,
            Quaternion.LookRotation(direction),
            Time.deltaTime * this.chasingRotSpeed);

        if (direction.magnitude > this.chasingAccuracy)
        {
            this.transform.Translate(0, 0, Time.deltaTime * this.chasingSpeed);
        }
    }

    #region Debugging
    /*
    void OnDrawGizmos()
    {
        if (circleCollider != null)
        {
            Gizmos.DrawWireSphere(transform.position, circleCollider.radius);
        }
    }

    void Update()
    {
        // target line
        Debug.DrawLine(rb2d.position, endPosition, Color.red);
    }
    */
    #endregion

}
