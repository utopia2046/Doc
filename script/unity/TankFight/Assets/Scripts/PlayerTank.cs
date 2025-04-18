using UnityEngine;

public class PlayerTank : MonoBehaviour
{
    public Transform targetTransform;
    public float targetDistanceTolerance = 3.0f;

    private float movementSpeed;
    private float rotationSpeed;

    // Use this for initialization
    void Start()
    {
        movementSpeed = 10.0f;
        rotationSpeed = 2.0f;
    }

    // Update is called once per frame
    void Update()
    {
        // check if the distance between tank & target is close enough
        if (Vector3.Distance(transform.position, targetTransform.position) < targetDistanceTolerance)
        {
            return;
        }

        Vector3 targetPosition = targetTransform.position;
        targetPosition.y = transform.position.y; // we don't enable gravity, so lock transform on current plane
        Vector3 direction = targetPosition - transform.position;

        // rotate and chase the target
        Quaternion tarRot = Quaternion.LookRotation(direction);
        transform.rotation = Quaternion.Slerp(transform.rotation, tarRot, rotationSpeed * Time.deltaTime);

        transform.Translate(new Vector3(0, 0, movementSpeed * Time.deltaTime));
    }
}
