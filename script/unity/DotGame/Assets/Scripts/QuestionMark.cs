using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuestionMark : MonoBehaviour
{
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.tag == "Player")
        {
            int randomscore = Random.Range(10, 101);
            Scoring.gameScore += randomscore;
            //Destroy(gameObject);
            gameObject.SetActive(false);
        }
    }
}
