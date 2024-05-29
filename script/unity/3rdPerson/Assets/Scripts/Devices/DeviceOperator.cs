using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeviceOperator : MonoBehaviour
{
    public float radius = 1.5f;  // how far away the player to activate devices

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            Collider[] hitColliders = Physics.OverlapSphere(transform.position, radius); // get nearby objects
            foreach (Collider collider in hitColliders)
            {
                Vector3 hitPosition = collider.transform.position;
                hitPosition.y = transform.position.y; // vertical direction won't matter
                Vector3 direction = hitPosition - transform.position;
                // if target object is 'facing' the device
                if (Vector3.Dot(transform.forward, direction.normalized) > .5f)
                {
                    // try to call Operate function on all nearby objects, regardless the target's type
                    collider.SendMessage("Operate", SendMessageOptions.DontRequireReceiver);
                }
            }
        }
    }
}
