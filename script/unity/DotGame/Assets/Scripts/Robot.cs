using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Robot : Monster
{
    new void Start()
    {
        base.Start();

        direction = Directions.Up;
        animator.Play("Robot_MoveUp");
    }

    new void FixedUpdate()
    {
        base.FixedUpdate();

        animator.SetFloat("Speed", rb.velocity.magnitude);
    }

    private void OnCollisionEnter2D(Collision2D collision)
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
