using UnityEngine;

public abstract class Character : MonoBehaviour
{
    public HitPoints hitPoints;
    public float maxHitPoints;
    public float startingHitPoints;

    public enum CharacterCategory
    {
        PLAYER,
        ENEMY
    }

    public CharacterCategory characterCategory;
}
