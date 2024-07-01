using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementController : MonoBehaviour
{
    public float movementSpeed = 3.0f;
    Vector2 movement = new Vector2();
    Rigidbody2D rb2D;

    // Start is called before the first frame update
    void Start()
    {
        rb2D = GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void Update()
    {

    }

    void FixedUpdate()
    {
        movement.x = Input.GetAxisRaw("Horizontal"); // 1 for right and -1 for left
        movement.y = Input.GetAxisRaw("Vertical");

        movement.Normalize(); // normalize when user move diagonally

        rb2D.velocity = movement * movementSpeed;
    }
}
