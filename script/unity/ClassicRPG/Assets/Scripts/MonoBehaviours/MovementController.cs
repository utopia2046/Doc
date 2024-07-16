using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementController : MonoBehaviour
{
    public float movementSpeed = 3.0f;
    Vector2 movement = new Vector2();
    Animator animator;
    string animationState = "AnimationState";
    Rigidbody2D rb2D;

    enum CharStates
    {
        idleSouth = 0,
        walkEast = 1,
        walkSouth = 2,
        walkWest = 3,
        walkNorth = 4
    }

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
        if (movement.x > 0)
        {
            animator.SetInteger(animationState, (int)CharStates.walkEast);
        }
        else if (movement.x < 0)
        {
            animator.SetInteger(animationState, (int)CharStates.walkWest);
        }
        else if (movement.y > 0)
        {
            animator.SetInteger(animationState, (int)CharStates.walkNorth);
        }
        else if (movement.y < 0)
        {
            animator.SetInteger(animationState, (int)CharStates.walkSouth);
        }
        else
        {
            animator.SetInteger(animationState, (int)CharStates.idleSouth);
        }
    }
}
