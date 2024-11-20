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

    new void OnCollisionEnter2D(Collision2D collision)
    {
        base.OnCollisionEnter2D(collision);

        // show animation depends on moving direction
        switch (direction)
        {
            case Directions.Up:
                animator.Play("Robot_MoveUp");
                break;
            case Directions.Right:
                animator.Play("Robot_MoveRight");
                break;
            case Directions.Down:
                animator.Play("Robot_MoveDown");
                break;
            case Directions.Left:
                animator.Play("Robot_MoveLeft");
                break;
        }
    }
}
