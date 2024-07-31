using System;
using UnityEngine;

[CreateAssetMenu(menuName = "Item")] // create an entry in Unity Editor Create submenu
public class Item : ScriptableObject
{
    public string objectName;
    public Sprite sprite;
    public int quantity;
    public bool stackable;

    public enum ItemType
    {
        COIN,
        HEALTH
    }

    public ItemType itemType;
}
