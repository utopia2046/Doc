using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Monster : MonoBehaviour
{
    public float speed;
    public Rigidbody2D rb;
    public Animator animator;
    protected Directions direction;
    protected Vector2 dirVector;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void FixedUpdate()
    {
        dirVector = Movement.GetDirectionVector(direction);
        rb.velocity = dirVector.normalized * speed;
        animator.SetFloat("Speed", rb.velocity.magnitude);
    }
}
