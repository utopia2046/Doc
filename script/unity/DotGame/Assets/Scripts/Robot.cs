using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Robot : MonoBehaviour
{
    public enum Directions
    {
        Up,
        Right,
        Down,
        Left
    }

    public float speed;
    private Rigidbody2D rb;
    public Animator animator;
    private Directions direction;
    private Vector2 dirVector;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        direction = Directions.Up;
        animator.Play("Robot_MoveUp");

        //Vector2 moveInput = new Vector2(0.0f, 1.0f); // initial move up
        //rb.velocity = moveInput.normalized * speed;
        //animator.SetFloat("Speed", rb.velocity.magnitude);
    }

    void FixedUpdate()
    {
        switch (direction)
        {
            case Directions.Up:
                dirVector = new Vector2(0.0f, 1.0f);
                break;
            case Directions.Down:
                dirVector = new Vector2(0.0f, -1.0f);
                break;
            case Directions.Left:
                dirVector = new Vector2(-1.0f, 0.0f);
                break;
            case Directions.Right:
                dirVector = new Vector2(1.0f, 0.0f);
                break;
        }

        rb.velocity = dirVector.normalized * speed;
        animator.SetFloat("Speed", rb.velocity.magnitude);
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        // retreat back a little to avoid stuck after turning
        Vector2 newPosition = new Vector2(
            transform.position.x - dirVector.x * 0.1f,
            transform.position.y - dirVector.y * 0.1f);
        rb.MovePosition(newPosition);
        //direction = (Directions)(((int)direction + 1) % 4);
        // on collision, turn clockwise and switch animation
        switch (direction)
        {
            case Directions.Up:
                direction = Directions.Right;
                animator.Play("Robot_MoveRight");
                break;
            case Directions.Right:
                direction = Directions.Down;
                animator.Play("Robot_MoveDown");
                break;
            case Directions.Down:
                direction = Directions.Left;
                animator.Play("Robot_MoveLeft");
                break;
            case Directions.Left:
                direction = Directions.Up;
                animator.Play("Robot_MoveUp");
                break;
        }
    }
}
