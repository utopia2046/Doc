using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Restart : MonoBehaviour
{
    public GameObject prefab;
    public Vector3 location;

    // Start is called before the first frame update
    void Start()
    {
        GenerateNewDonut();
    }

    // Update is called once per frame
    void Update()
    {
        //if (Input.GetMouseButtonDown(0))
        //{
        //    Debug.Log("Restart clicked");
        //    GenerateNewDonut();
        //}
    }

    void OnMouseDown()
    {
        Debug.Log("Restart clicked");
        if (!GameState.inBox)
        {
            GameState.state = GameState.gamePlay;
            GameState.inBox = false;
            Scoring.gameScore = 0;
            GenerateNewDonut();
        }
    }

    void GenerateNewDonut()
    {
        Instantiate(prefab, location, Quaternion.identity);
    }
}
