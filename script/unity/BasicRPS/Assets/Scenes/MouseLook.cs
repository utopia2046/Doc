using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseLook : MonoBehaviour
{
    public enum RotationAxes
    {
        MouseXAndY = 0,
        MouseX = 1,
        MouseY = 2,
    }

    public RotationAxes axes = RotationAxes.MouseX;
    public float sensitivityHor = 9.0f;
    public float sensitivityVer = 3.0f;
    public float maxVert = 45.0f;
    public float minVert = -45.0f;
    private float verticalRot = 0;

    // Start is called before the first frame update
    void Start()
    {
        Rigidbody body = GetComponent<Rigidbody>();
        if (body != null)
        {
            body.freezeRotation = true; // Disable player's rigidbody rotation
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (axes == RotationAxes.MouseX) // horizontal rotate
        {
            this.transform.Rotate(0, sensitivityHor * Input.GetAxis("Mouse X"), 0);
        }
        else if (axes == RotationAxes.MouseY)
        {
            verticalRot -= sensitivityVer * Input.GetAxis("Mouse Y");
            verticalRot = Mathf.Clamp(verticalRot, minVert, maxVert);
            float horizontalRot = this.transform.localEulerAngles.y;

            this.transform.localEulerAngles = new Vector3(verticalRot, horizontalRot, 0);
        }
        else
        {
            verticalRot -= sensitivityVer * Input.GetAxis("Mouse Y");
            verticalRot = Mathf.Clamp(verticalRot, minVert, maxVert);

            float delta = sensitivityHor * Input.GetAxis("Mouse X");
            float horizontalRot = this.transform.localEulerAngles.y + delta;

            this.transform.localEulerAngles = new Vector3(verticalRot, horizontalRot, 0);
        }
    }
}
