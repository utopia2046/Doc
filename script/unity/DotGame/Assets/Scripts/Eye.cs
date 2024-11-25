using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Eye : Monster
{
    new void Start()
    {
        base.Start();

        direction = Directions.Up;
        animator.Play("Eye_Move");
    }

    new void FixedUpdate()
    {
        base.FixedUpdate();
    }

    new void OnCollisionEnter2D(Collision2D collision)
    {
        base.OnCollisionEnter2D(collision);
    }
}
