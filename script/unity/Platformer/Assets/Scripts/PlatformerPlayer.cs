using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlatformerPlayer : MonoBehaviour
{
    public float speed = 4.5f;

    private Rigidbody2D body;
    private Animator anim;

    // Start is called before the first frame update
    void Start()
    {
        body = GetComponent<Rigidbody2D>(); // get Rigidbody2D component instance
        anim = GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
        // set only horizontal movement, preserve preexisting vertical movement
        float deltaX = Input.GetAxis("Horizontal") * speed;
        Vector2 movement = new Vector2(deltaX, body.velocity.y);
        body.velocity = movement; // moving Transform.position will ignore collision detection

        anim.SetFloat("speed", Mathf.Abs(deltaX));
        if (!Mathf.Approximately(deltaX, 0))
        {
            // when moving, scale positive or negative 1 to face right or left
            transform.localScale = new Vector3(Mathf.Sign(deltaX), 1, 1);
        }
    }
}
