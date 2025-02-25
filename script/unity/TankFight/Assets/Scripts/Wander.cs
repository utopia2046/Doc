using UnityEngine;

public class Wander : MonoBehaviour
{
    private Vector3 targetPosition;

    private float movementSpeed = 5.0f;
    private float rotationSpeed = 2.0f;
    private float targetPositionTolerance = 3.0f;
    private float minX;
    private float maxX;
    private float minZ;
    private float maxZ;

    void Start()
    {
        minX = -45.0f;
        maxX = 45.0f;

        minZ = -45.0f;
        maxZ = 45.0f;

        //Get Wander Position
        GetNextPosition();
    }

    void Update()
    {
        if (Vector3.Distance(targetPosition, transform.position) <= targetPositionTolerance)
        {
            // if already near current target, get next target
            GetNextPosition();
        }

        Quaternion targetRotation = Quaternion.LookRotation(targetPosition - transform.position);
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);

        transform.Translate(new Vector3(0, 0, movementSpeed * Time.deltaTime));
    }

    void GetNextPosition()
    {
        // get a random position in fence
        targetPosition = new Vector3(Random.Range(minX, maxX), 0.5f, Random.Range(minZ, maxZ));
    }
}
