using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Arrow : MonoBehaviour
{
    public float speed;
    private Directions direction;
    private Vector2 dirVector;
    private Rigidbody2D rb;
    AudioSource whoosh;
    //AudioSource bounce;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        // direction = Directions.Up;
        // rb.MoveRotation(90.0f);
        // Debug.Log("Arrow.speed = " + speed + "; Arrow.direction = " + direction);
        //AudioSource[] audios = GetComponents<AudioSource>();
        whoosh = GetComponent<AudioSource>();//audios[0];
        //bounce = audios[1];
        whoosh.Play();
    }

    public void SetDirection(Directions dir)
    {
        this.direction = dir;
    }

    private void FixedUpdate()
    {
        dirVector = Movement.GetDirectionVector(direction);
        rb.velocity = dirVector.normalized * speed;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        // Ignore if hit player
        if ((collision.gameObject.tag == "Player") || (collision.gameObject.tag == "Arrow"))
        {
            Physics2D.IgnoreCollision(collision.collider, gameObject.GetComponent<Collider2D>());
            return;
        }

        // kill eye
        if (collision.gameObject.name == "Eye")
        {
            Debug.Log("Kill an eye");
            Scoring.gameScore += 50;
            var eye = collision.gameObject.GetComponent<Eye>();
            eye.isDying = true;
            eye.PlaySound(collision.gameObject);
            //collision.gameObject.SetActive(false);
            //Destroy(collision.gameObject);
        }

        // hit robot, it grunts but won't die
        if (collision.gameObject.name == "Robot")
        {
            Debug.Log("Hit a robot");
            var robot = collision.gameObject.GetComponent<Robot>();
            robot.PlaySound(collision.gameObject);
            //collision.gameObject.SetActive(false);
            //Destroy(collision.gameObject);
        }

        // disable arrow
        //Destroy(gameObject);
        //bounce.Play();
        gameObject.SetActive(false);
    }
}
