using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlatformerPlayer : MonoBehaviour
{
    public float speed = 4.5f;
    public float jumpForce = 12.0f;

    private Rigidbody2D body;
    private BoxCollider2D box;
    private Animator anim;

    // Start is called before the first frame update
    void Start()
    {
        body = GetComponent<Rigidbody2D>(); // get Rigidbody2D component instance
        box = GetComponent<BoxCollider2D>();
        anim = GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
        // set only horizontal movement, preserve preexisting vertical movement
        float deltaX = Input.GetAxis("Horizontal") * speed;
        Vector2 movement = new Vector2(deltaX, body.velocity.y);
        body.velocity = movement; // moving Transform.position will ignore collision detection

        Vector3 max = box.bounds.max;
        Vector3 min = box.bounds.min;
        Vector2 corner1 = new Vector2(max.x, min.y - .1f); // a little below Player's right foot corner
        Vector2 corner2 = new Vector2(min.x, min.y - .2f); // a little below Player's left foot corner
        Collider2D hit = Physics2D.OverlapArea(corner1, corner2); // check if any collider overlapping under Player's feet
        bool grounded = (hit != null);  // detect if the player is on the ground

        // add an upward force to jump if spacebar is pressed
        if (grounded && Input.GetKeyDown(KeyCode.Space))
        {
            body.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
        }

        anim.SetFloat("speed", Mathf.Abs(deltaX));
        if (!Mathf.Approximately(deltaX, 0))
        {
            // when moving, scale positive or negative 1 to face right or left
            transform.localScale = new Vector3(Mathf.Sign(deltaX), 1, 1);
        }
    }
}
