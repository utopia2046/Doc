using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Monster : MonoBehaviour
{
    public float speed;
    public Animator animator;
    protected Rigidbody2D rb;
    protected Directions direction;
    protected Vector2 dirVector;

    public void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    public void FixedUpdate()
    {
        dirVector = Movement.GetDirectionVector(direction);
        rb.velocity = dirVector.normalized * speed;
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        // retreat back a little to avoid stuck after turning
        Vector2 newPosition = new Vector2(
            transform.position.x - dirVector.x * 0.1f,
            transform.position.y - dirVector.y * 0.1f);
        rb.MovePosition(newPosition);

        // on collision, turn clockwise and switch animation
        switch (direction)
        {
            case Directions.Up:
                direction = Directions.Right;
                break;
            case Directions.Right:
                direction = Directions.Down;
                break;
            case Directions.Down:
                direction = Directions.Left;
                break;
            case Directions.Left:
                direction = Directions.Up;
                break;
        }
    }
}
