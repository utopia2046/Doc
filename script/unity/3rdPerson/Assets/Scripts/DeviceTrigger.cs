using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeviceTrigger : MonoBehaviour
{
    [SerializeField] GameObject[] targets;

    void OnTriggerEnter(Collider other)
    {
        Debug.Log("Enter Door Trigger");
        foreach (GameObject target in targets)
        {
            target.SendMessage("Activate");
        }
    }

    void OnTriggerExit(Collider other)
    {
        Debug.Log("Exit Door Trigger");
        foreach (GameObject target in targets)
        {
            target.SendMessage("Deactivate");
        }
    }
}
