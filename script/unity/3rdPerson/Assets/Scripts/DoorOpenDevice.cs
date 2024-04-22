using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DoorOpenDevice : MonoBehaviour
{
    [SerializeField] Vector3 dPos;  // amount to offset the position by when the door opens

    private bool open;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

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
}
