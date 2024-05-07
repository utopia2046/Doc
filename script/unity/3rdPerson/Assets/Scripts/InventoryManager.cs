using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InventoryManager : MonoBehaviour, IGameManager
{
    public ManagerStatus status { get; private set; }

    private Dictionary<string, int> items;

    public void Startup()
    {
        Debug.Log("Inventory manager starting...");
        status = ManagerStatus.Initializing;
        items = new Dictionary<string, int>();
        status = ManagerStatus.Started;
    }

    public void AddItem(string name)
    {
        if (items.ContainsKey(name))
        {
            items[name] += 1;
        }
        else
        {
            items[name] = 1;
        }

        DisplayItems();
    }

    private void DisplayItems()
    {
        string itemDisplay = "Items: ";

        foreach (KeyValuePair<string, int> item in items)
        {
            itemDisplay += $"{item.Key}({item.Value}) ";
        }

        Debug.Log(itemDisplay);
    }
}
