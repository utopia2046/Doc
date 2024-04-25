using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DoorOpenDevice : MonoBehaviour
{
    [SerializeField] Vector3 dPos;  // amount to offset the position by when the door opens

    private bool open;

    public void Operate()
    {
        if (open)
        {
            Vector3 pos = transform.position - dPos;
            transform.position = pos;
        }
        else
        {
            Vector3 pos = transform.position + dPos;
            transform.position = pos;
        }
        open = !open;
    }

    public void Activate()
    {
        Debug.Log("Activate called");
        if (!open)
        {
            Vector3 pos = transform.position + dPos;
            transform.position = pos;
            open = true;
        }
    }

    public void Deactivate()
    {
        Debug.Log("Deactivate called");
        if (open)
        {
            Vector3 pos = transform.position - dPos;
            transform.position = pos;
            open = false;
        }
    }
}
