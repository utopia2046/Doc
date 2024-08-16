using System.Collections;
using UnityEngine;

public class Enemy : Character
{
    // Enemy has no health bar, so we only need a simple float to save it's HP
    float hitPoints;
    public int damageStrength;

    Coroutine damageCoroutine;

    private void OnEnable()
    {
        ResetCharacter();
    }

    public override void ResetCharacter()
    {
        hitPoints = startingHitPoints;
    }

    public override IEnumerator DamageCharacter(int damage, float interval)
    {
        while (true)
        {
            hitPoints = hitPoints - damage;

            if (hitPoints <= float.Epsilon)
            {
                KillCharacter();
                break;
            }

            if (interval > float.Epsilon)
            {
                yield return new WaitForSeconds(interval);
            }
            else
            {
                break;
            }
        }
    }

    void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Player"))
        {
            Player player = collision.gameObject.GetComponent<Player>();

            // only call DamageCharacter on the player if we don't currently have a DamageCharacter() Coroutine running.
            if (damageCoroutine == null)
            {
                damageCoroutine = StartCoroutine(player.DamageCharacter(damageStrength, 1.0f));
            }
        }
    }

    void OnCollisionExit2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Player"))
        {
            if (damageCoroutine != null)
            {
                StopCoroutine(damageCoroutine);
                damageCoroutine = null;
            }
        }
    }
}
