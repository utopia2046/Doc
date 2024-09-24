using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Scoring : MonoBehaviour
{
    public int fontSize = 30;
    public Rect scoreRect = new Rect(20, 20, 200, 50);
    public string prefix = "Score: ";
    public static int gameScore;

    void Start()
    {
        gameScore = 0;
    }

    void Update()
    {
    }

    private void OnGUI()
    {
        GUI.skin.box.fontSize = fontSize;
        GUI.Box(scoreRect, prefix + gameScore.ToString());
    }
}
