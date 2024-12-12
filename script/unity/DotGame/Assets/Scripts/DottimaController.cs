using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class DottimaController : MonoBehaviour
{
    public float speed;
    public Vector3 initLocation;
    public GameObject shot;
    public Animator animator;
    public float levelCompleteTimer = 2.0f;
    private Rigidbody2D rb;
    private Directions direction;
    private float zRot;
    private Arrow arrow;
    private GameObject bomb = null;
    private float deathTimer = 1.0f;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        direction = Directions.Up;
    }

    private void FixedUpdate()
    {
        if (GameState.state != GameState.PLAYING)
        {
            return;
        }

        Vector2 moveInput = new Vector2(
            Input.GetAxisRaw("Horizontal"),
            Input.GetAxisRaw("Vertical"));
        rb.velocity = moveInput.normalized * speed;
        animator.SetFloat("Speed", rb.velocity.magnitude); // set animator parameter using speed magnitude
    }

    private void Update()
    {
        if (GameState.state == GameState.LEVELCOMPLETE)
        {
            rb.velocity = Vector3.zero;
            levelCompleteTimer -= Time.deltaTime;
            if (levelCompleteTimer < 0.0f)
            {
                GameState.level++;
                Debug.Log("Loading scene #" + GameState.level.ToString());
                GameState.state = GameState.PLAYING;
                SceneManager.LoadScene(GameState.level);
            }
            return;
        }

        float x, y;
        x = rb.velocity.x;
        y = rb.velocity.y;

        if (GameState.state == GameState.GAMEOVER) // && (deathTimer > 0))
        {
            Debug.Log("Dead Dottima");
            float shrink = 1.0f - 2.0f * Time.deltaTime;
            float rotSpeed = -400.0f * Time.deltaTime;
            rb.velocity = Vector2.zero;
            rb.rotation += rotSpeed;
            transform.localScale = new Vector3(
                transform.localScale.x * shrink,
                transform.localScale.y * shrink,
                transform.localScale.z);
            deathTimer -= Time.deltaTime;
            return;
        }

        // get player's moving direction
        if (x != 0 || y != 0)
        {
            //Debug.Log("x = " + x + "; y = " + y);
            if (Math.Abs(x) > Math.Abs(y))
            {
                direction = (x < 0) ? Directions.Left : Directions.Right;
            }
            else
            {
                direction = (y > 0) ? Directions.Up : Directions.Down;
            }
            //Debug.Log("Direction: " + direction);
        }

        // on space bar hitting, shot an arrow
        if (Input.GetKeyDown("space"))
        {
            // drop bomb
            if (bomb != null)
            {
                Debug.Log("Drop a bomb");
                bomb.GetComponent<Bomb>().Use();
                bomb.transform.SetParent(null);
                bomb = null;
                return;
            }

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

    private void OnCollisionEnter2D(Collision2D collision)
    {
        string tag = collision.gameObject.tag;

        switch (tag)
        {
            case "Bomb":
                if (bomb == null)
                {
                    Debug.Log("Pick up bomb");
                    bomb = collision.gameObject;
                    bomb.transform.SetParent(gameObject.transform);
                    bomb.transform.localPosition = new Vector3(-0.2f, 0.2f, -1.0f);
                    Physics2D.IgnoreCollision(
                        collision.collider,
                        gameObject.GetComponent<Collider2D>()
                    );
                }
                break;
            case "Enemy":
                Debug.Log("Hit an enemy");
                Scoring.lives--;
                Scoring.gameScore -= 100;
                Debug.Log("Loose a life, remaining lives: " + Scoring.lives);
                if (Scoring.lives <= 0)
                {
                    Debug.Log("Game over");
                    GameState.state = GameState.GAMEOVER;
                    GameState.stateText = "Game Over";
                    Scoring.gameScore = 0;
                    return;
                }
                gameObject.transform.position = initLocation;
                break;
            case "Exit":
                GameState.state = GameState.LEVELCOMPLETE;
                GameState.stateText = "Level Complete";
                Scoring.gameScore += 500;
                break;
            case "Blockade":
                GameState.state = GameState.GAMECOMPLETE;
                GameState.stateText = "THE END";
                Scoring.gameScore += 1000;
                break;
        }
    }
}
