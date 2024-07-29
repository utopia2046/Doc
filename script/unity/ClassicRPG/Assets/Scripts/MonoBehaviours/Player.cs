using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : Character
{
    public HealthBar healthBarPrefab;
    HealthBar healthBar;

    // Start is called before the first frame update
    void Start()
    {
        hitPoints.value = startingHitPoints;
        healthBar = Instantiate(healthBarPrefab);
        healthBar.character = this;
    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if (collision.gameObject.CompareTag("CanBePickedUp"))
        {
            Item hitObject = collision.gameObject.GetComponent<Consumable>().item;

            if (hitObject != null)
            {
                print("Hit: " + hitObject.objectName);
                bool shouldDisappear = false;

                switch (hitObject.type)
                {
                    case Item.ItemType.COIN:
                        shouldDisappear = true;
                        break;
                    case Item.ItemType.HEALTH:
                        shouldDisappear = AdjustHitPoints(hitObject.quantity);
                        break;
                    default:
                        break;
                }

                if (shouldDisappear)
                {
                    Debug.Log("Item collected");
                    collision.gameObject.SetActive(false); // hide the collected item
                }
            }
        }
    }

    public bool AdjustHitPoints(int amount)
    {
        if (hitPoints.value < maxHitPoints && hitPoints.value > 0)
        {
            float newValue = (hitPoints.value + amount < 0) ? 0 : (hitPoints.value + amount);
            hitPoints.value = (newValue > maxHitPoints) ? maxHitPoints : newValue;
            print("Adjusted HP by: " + amount + ". New value: " + hitPoints.value);
            return true;
        }

        return false;
    }
}
