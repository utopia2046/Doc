using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MemoryCard : MonoBehaviour
{
    [SerializeField] GameObject cardBack;

    public void OnMouseDown()
    {
        Debug.Log("card 1 clicked");
        if (cardBack.activeSelf)
        {
            cardBack.SetActive(false);
        }
    }
}
