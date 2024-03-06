using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(CharacterController))]
[AddComponentMenu("Control Script/FPS Input")]
public class FPSInput : MonoBehaviour
{
    public float speed = 0.5f;
    public float gravity = 1f;
    private CharacterController charController;

    // Start is called before the first frame update
    void Start()
    {
        charController = GetComponent<CharacterController>(); // GetComponent to get in editor attached component by type
    }

    // Update is called once per frame
    void Update()
    {
        float deltaX = speed * Input.GetAxis("Horizontal");
        float deltaZ = speed * Input.GetAxis("Vertical");
        Vector3 movement = new Vector3(deltaX, 0, deltaZ);

        movement = Vector3.ClampMagnitude(movement, speed); // limit diagonal movement to the same speed
        movement *= Time.deltaTime;                         // move frame-rate independently
        //movement.y = gravity;
        //movement = transform.TransformDirection(movement);  // tranform movement vector from local to global coordinates
        Debug.Log(string.Format("movement x = {0}, y = {1}, z= {2}", movement.x, movement.y, movement.z));

        transform.Translate(deltaX * Time.deltaTime, 0, deltaZ * Time.deltaTime);
        //charController.Move(movement);                      // use CharacterController move instead of transform to enable collition detection
    }
}
