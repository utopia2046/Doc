using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MemoryCard : MonoBehaviour
{
    [SerializeField] GameObject cardBack; // attribute SerializeField enable object reference setting in Editor UI
    [SerializeField] SceneController controller;

    private int _id;
    public int Id
    {
        get
        {
            return _id;
        }
    }

    public void SetCard(int id, Sprite image)
    {
        _id = id;
        GetComponent<SpriteRenderer>().sprite = image;
    }

    public void OnMouseDown()
    {
        Debug.Log(string.Format("card id = {0} clicked", _id));
        if (cardBack.activeSelf)
        {
            cardBack.SetActive(false); // hide the card_back object on top to reveal card undereath
        }
    }

    public void Unreveal()
    {
        cardBack.SetActive(true);
    }
}
