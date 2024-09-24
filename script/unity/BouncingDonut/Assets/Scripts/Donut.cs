using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Donut : MonoBehaviour
{
    private void OnCollisionEnter2D(Collision2D other)
    {
        if (other != null && other.gameObject != null && other.gameObject.tag == "WoodPlank")
        {
            Scoring.gameScore += 10;
        }
    }
}
