using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Robot : MonoBehaviour
{
    public enum Directions
    {
        Up,
        Down,
        Left,
        Right
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
}
