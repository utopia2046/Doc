using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DottimaController : MonoBehaviour
{
    public float speed;
    public Animator animator;
    private Rigidbody2D rb;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    private void FixedUpdate()
    {
        Vector2 moveInput = new Vector2(
            Input.GetAxisRaw("Horizontal"),
            Input.GetAxisRaw("Vertical"));
        rb.velocity = moveInput.normalized * speed;
        animator.SetFloat("Speed", rb.velocity.magnitude); // set animator parameter using speed magnitude
    }
}
