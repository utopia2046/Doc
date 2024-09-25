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
        // Draw Score on Top Left corner
        GUI.skin.box.fontSize = fontSize;
        GUI.Box(scoreRect, prefix + gameScore.ToString());
        // Draw Game Over
        if (GameState.state == GameState.gameOver)
        {
            GUI.skin.box.fontSize = 60;
            GUI.Box(new Rect(
                Screen.width / 2 - 200,
                Screen.height / 2 - 40,
                400,
                80),
            GameState.inBox ? "Success" : "Fail");
        }
    }
}
