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

    public RotationAxes Axes = RotationAxes.MouseX;
    public float SensitivityHor = 9.0f;
    public float SensitivityVer = 3.0f;
    public float MaxVert = 45.0f;
    public float MinVert = -45.0f;
    private float verticalRot = 0;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Axes == RotationAxes.MouseX) // horizontal rotate
        {
            this.transform.Rotate(0, SensitivityHor * Input.GetAxis("Mouse X"), 0);
        }
        else if (Axes == RotationAxes.MouseY)
        {
            verticalRot -= SensitivityVer * Input.GetAxis("Mouse Y");
            verticalRot = Mathf.Clamp(verticalRot, MinVert, MaxVert);
            float horizontalRot = this.transform.localEulerAngles.y;

            this.transform.localEulerAngles = new Vector3(verticalRot, horizontalRot, 0);
        }
        else
        {
            this.transform.Rotate(SensitivityVer * Input.GetAxis("Mouse Y"), SensitivityHor * Input.GetAxis("Mouse X"), 0);
        }
    }
}
