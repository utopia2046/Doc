using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Scoring : MonoBehaviour
{
    const int fontSize = 24;
    public static int gameScore = 0;
    public static int lives = 3;

    //private void Awake()
    //{
    //    DontDestroyOnLoad(gameObject);
    //}

    //*
    private void OnGUI()
    {
        // Draw Score on Top Left corner
        GUI.skin.box.fontSize = fontSize;
        Rect scoreRect = new Rect(10, 10, 360, 40);
        string levelText = "Level: " + GameState.level.ToString();
        string scoreText = "Score: " + gameScore.ToString();
        string livesText = "Lives: " + lives.ToString();
        GUI.Box(scoreRect, levelText + " " + scoreText + " " + livesText);

        // Draw Game Over
        if (GameState.state == GameState.GAMEOVER || GameState.state == GameState.LEVELCOMPLETE)
        {
            GUIStyle fontStyle = new GUIStyle();
            fontStyle.fontSize = fontSize;
            fontStyle.normal.textColor = Color.white;
            string text = "Game Over";
            if (GUI.Button(new Rect(
                Screen.width / 2 - 50,
                Screen.height / 2 - 40,
                100,
                40),
                text,
                fontStyle))
            {
                //GameState.state = GameState.PLAYING;
            }
        }
    }
    //*/
}
