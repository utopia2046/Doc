using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementController : MonoBehaviour
{
    public float movementSpeed = 3.0f;
    Vector2 movement = new Vector2();
    Animator animator;
    Rigidbody2D rb2D;

    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponent<Animator>();
        rb2D = GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void Update()
    {
        UpdateState();
    }

    void FixedUpdate()
    {
        MoveCharacter();
    }

    void MoveCharacter()
    {
        movement.x = Input.GetAxisRaw("Horizontal"); // 1 for right and -1 for left
        movement.y = Input.GetAxisRaw("Vertical");

        movement.Normalize(); // normalize when user move diagonally

        rb2D.velocity = movement * movementSpeed;
    }

    void UpdateState()
    {
        if (Mathf.Approximately(movement.x, 0) && Mathf.Approximately(movement.y, 0))
        {
            animator.SetBool("isWalking", false);
        }
        else
        {
            animator.SetBool("isWalking", true);
        }

        animator.SetFloat("xDir", movement.x);
        animator.SetFloat("yDir", movement.y);
    }
}
