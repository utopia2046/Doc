using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuestionMark : MonoBehaviour
{
    public float disappearTime = 1.0f;
    public float disappearSpeed = 1.0f;
    private AudioSource ding;
    private float deathTimer;

    void Start()
    {
        ding = GetComponent<AudioSource>();
        deathTimer = disappearTime;
    }

    void Update()
    {
        if (deathTimer < disappearTime)
        {
            deathTimer -= Time.deltaTime;
            float shrink = 1.0f - disappearSpeed * Time.deltaTime;
            transform.localScale = new Vector3(
                shrink * transform.localScale.x,
                shrink * transform.localScale.y,
                transform.localScale.z);
            if (deathTimer < 0.0f)
            {
                gameObject.SetActive(false);
            }
        }
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (deathTimer < disappearTime)
        {
            Physics2D.IgnoreCollision(collision.collider, gameObject.GetComponent<Collider2D>());
            return;
        }

        if (collision.gameObject.tag == "Player")
        {
            int randomscore = Random.Range(10, 101);
            Scoring.gameScore += randomscore;
            ding.Play();
            //gameObject.SetActive(false);
            deathTimer -= 0.01f;
        }
    }
}
