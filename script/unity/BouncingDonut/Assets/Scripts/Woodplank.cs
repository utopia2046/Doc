using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Woodplank : MonoBehaviour
{
    void Update()
    {
        if (GameState.state != GameState.gamePlay)
        {
            return;
        }

        if (Input.GetKey("w") || Input.GetKey("up"))
        {
            transform.Translate(0, 5 * Time.deltaTime, 0);
        }
        else if (Input.GetKey("s") || Input.GetKey("down"))
        {
            transform.Translate(0, -5 * Time.deltaTime, 0);
        }
    }
}
