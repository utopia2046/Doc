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
        direction = Directions.Up;
        rb.MoveRotation(90.0f);
    }

    private void FixedUpdate()
    {
        dirVector = Movement.GetDirectionVector(direction);
        rb.velocity = dirVector.normalized * speed;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        // Ignore if hit player
        if (collision.gameObject.tag == "Player")
        {
            Physics2D.IgnoreCollision(collision.collider, gameObject.GetComponent<Collider2D>());
            return;
        }

        Vector2 newPosition = Movement.RetreatALittle(transform.position, dirVector);
        rb.MovePosition(newPosition);

        speed = 0f;
    }
}
