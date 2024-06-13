using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerManager : MonoBehaviour, IGameManager
{
    public ManagerStatus status { get; private set; }
    public int health { get; private set; }
    public int maxHealth { get; private set; }

    public void Startup()
    {
        status = ManagerStatus.Initializing;
        Debug.Log("Player manager starting...");

        maxHealth = 100;
        health = 50;

        status = ManagerStatus.Started;
    }

    public void UpdateData(int health, int maxHealth)
    {
        this.health = health;
        this.maxHealth = maxHealth;
    }

    public void ChangeHealth(int value)
    {
        health += value;
        health = (health > maxHealth) ? maxHealth : health;
        health = (health < 0) ? 0 : health;
        Debug.Log($"Health: {health}/{maxHealth}");
        Messenger.Broadcast(GameEvent.HEALTH_UPDATED);
    }
}
