using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DottimaController : MonoBehaviour
{
    public float speed;
    public GameObject shot;
    public Animator animator;
    private Rigidbody2D rb;
    private Directions direction;
    private float zRot;
    private Arrow arrow;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        direction = Directions.Up;
    }

    private void FixedUpdate()
    {
        Vector2 moveInput = new Vector2(
            Input.GetAxisRaw("Horizontal"),
            Input.GetAxisRaw("Vertical"));
        rb.velocity = moveInput.normalized * speed;
        animator.SetFloat("Speed", rb.velocity.magnitude); // set animator parameter using speed magnitude
    }

    private void Update()
    {
        float x, y;
        x = rb.velocity.x;
        y = rb.velocity.y;

        // get player's moving direction
        if (x != 0 || y != 0)
        {
            Debug.Log("x = " + x + "; y = " + y);
            if (Math.Abs(x) > Math.Abs(y))
            {
                direction = (x < 0) ? Directions.Left : Directions.Right;
            }
            else
            {
                direction = (y > 0) ? Directions.Up : Directions.Down;
            }
            Debug.Log("Direction: " + direction);
        }

        // on space bar hitting, shot an arrow
        if (Input.GetKeyDown("space"))
        {
            // arrow direction is player's moving direction
            switch (direction)
            {
                case Directions.Left:
                    zRot = 0f;
                    break;
                case Directions.Right:
                    zRot = 180f;
                    break;
                case Directions.Up:
                    zRot = -90f;
                    break;
                case Directions.Down:
                    zRot = 90f;
                    break;
            }

            // create an arrow
            GameObject ar = Instantiate(
                shot,
                new Vector3(transform.position.x, transform.position.y, 1.0f),
                Quaternion.Euler(0, 0, zRot)
            );

            // set arrow's direction and speed
            ar.GetComponent<Arrow>().SetDirection(direction);
            if (x != 0 || y != 0)
            {
                ar.GetComponent<Arrow>().speed += speed;
            }
        }
    }
}
