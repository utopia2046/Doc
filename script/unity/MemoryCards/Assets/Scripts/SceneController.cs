using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController : MonoBehaviour
{
    public const int gridRows = 2;
    public const int gridCols = 4;
    public const float offsetX = 2f;
    public const float offsetY = 2.5f;

    [SerializeField] MemoryCard originalCard;
    [SerializeField] Sprite[] images;

    private int[] ShuffleArray(int[] numbers)
    {
        int[] newArray = numbers.Clone() as int[];

        for (int i = 0; i < newArray.Length; i++)
        {
            int tmp = newArray[i];
            int r = UnityEngine.Random.Range(i, newArray.Length);
            newArray[i] = newArray[r];
            newArray[r] = tmp;
        }

        return newArray;
    }

    // Start is called before the first frame update
    void Start()
    {
        Vector3 startPos = originalCard.transform.position;  // get first card position

        int[] numbers = { 0, 0, 1, 1, 2, 2, 3, 3 };
        numbers = ShuffleArray(numbers);

        Debug.Log("images.Length = " + images.Length.ToString());
        for (int i = 0; i < gridCols; i++)
        {
            for (int j = 0; j < gridRows; j++)
            {
                MemoryCard card;
                if (i == 0 && j == 0)
                {
                    card = originalCard;
                }
                else
                {
                    card = Instantiate(originalCard) as MemoryCard; // copy original card
                }

                //int id = UnityEngine.Random.Range(0, images.Length);
                int index = j * gridCols + i;
                int id = numbers[index];
                Debug.Log(String.Format("i = {0}, j = {1}, id = {2}", i, j, id));
                originalCard.SetCard(id, images[id]);

                float posX = offsetX * i + startPos.x;
                float posY = -offsetY * j + startPos.y;
                card.transform.position = new Vector3(posX, posY, startPos.z); // set x and y, keep z the same as 1st card
                Debug.Log(String.Format("Position card[{0}, {1}] at X = {2}, Y = {3}", i, j, posX, posY));
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
