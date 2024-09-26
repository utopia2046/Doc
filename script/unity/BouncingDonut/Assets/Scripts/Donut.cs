using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Donut : MonoBehaviour
{
    public Rigidbody2D rb;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        if (GameState.state == GameState.gameOver)
        {
            Debug.Log("Game Over");
            rb.velocity = Vector2.zero;
        }
    }

    private void OnCollisionEnter2D(Collision2D other)
    {
        if (other != null && other.gameObject != null)
        {
            Debug.Log("Donut hit object with tag: " + other.gameObject.tag);

            switch (other.gameObject.tag)
            {
                case "WoodPlank":
                    Scoring.gameScore += 10;
                    GameState.inBox = false;
                    break;
                case "Sphere":
                    Scoring.gameScore += 50;
                    GameState.inBox = false;
                    break;
                case "DonutBox":
                    Scoring.gameScore += 100;
                    if (!GameState.inBox)
                    {
                        // start 5s counting down when donut first hit goal
                        StartCoroutine(WaitFor5Secs());
                    }
                    break;
                case "Floor":
                    Scoring.gameScore = 0;
                    GameState.state = GameState.gameOver;
                    GameState.inBox = false;
                    Debug.Log("Fail");
                    gameObject.SetActive(false);
                    break;
            }
        }
    }

    IEnumerator WaitFor5Secs()
    {
        GameState.inBox = true;
        Debug.Log("Start WaitFor5Secs Coroutine: " + Time.time);
        yield return new WaitForSeconds(5);
        Debug.Log("End WaitFor5Secs Coroutine: " + Time.time);
        if (GameState.inBox)
        {
            GameState.state = GameState.gameOver;
            Debug.Log("Success");
        }
    }
}
