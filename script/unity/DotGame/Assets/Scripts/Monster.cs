using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Monster : MonoBehaviour
{
    public float speed;
    public Animator animator;

    protected Rigidbody2D rb;
    protected Directions direction;
    protected Vector2 dirVector;

    private float deathTimer;
    public bool isDying;

    public void Start()
    {
        deathTimer = 1.0f;
        isDying = false;
        rb = GetComponent<Rigidbody2D>();
    }

    public void FixedUpdate()
    {
        if (isDying)
        {
            rb.velocity = Vector3.zero;
            return;
        }

        dirVector = Movement.GetDirectionVector(direction);
        rb.velocity = dirVector.normalized * speed;
    }

    public void Update()
    {
        if (isDying)
        {
            deathTimer -= Time.deltaTime;
        }
        if (deathTimer < 0.0f)
        {
            gameObject.SetActive(false);
        }
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        // retreat back a little to avoid stuck after turning
        Vector2 newPosition = Movement.RetreatALittle(transform.position, dirVector);
        rb.MovePosition(newPosition);

        // on collision, turn clockwise and switch animation
        switch (direction)
        {
            case Directions.Up:
                direction = Directions.Right;
                break;
            case Directions.Right:
                direction = Directions.Down;
                break;
            case Directions.Down:
                direction = Directions.Left;
                break;
            case Directions.Left:
                direction = Directions.Up;
                break;
        }
    }

    public void PlaySound(GameObject obj)
    {
        AudioSource sound = obj.GetComponent<AudioSource>();
        Debug.Log(obj.name + " starts screaming");
        if (sound != null)
        {
            sound.Play();
        }
    }
}
