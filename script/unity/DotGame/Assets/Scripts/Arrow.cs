using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Arrow : MonoBehaviour
{
    public float speed;
    private Directions direction;
    private Vector2 dirVector;
    private Rigidbody2D rb;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        // direction = Directions.Up;
        // rb.MoveRotation(90.0f);
        Debug.Log("Arrow.speed = " + speed + "; Arrow.direction = " + direction);
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

        if (collision.gameObject.tag == "Eye")
        {
            collision.gameObject.SetActive(false);
            //Destroy(collision.gameObject);
        }

        //Destroy(gameObject);
        gameObject.SetActive(false);

        //Vector2 newPosition = Movement.RetreatALittle(transform.position, dirVector);
        //rb.MovePosition(newPosition);

        //speed = 0f;
    }
}
