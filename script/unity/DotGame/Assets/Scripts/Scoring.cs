using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

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
        if (GameState.state != GameState.PLAYING)
        {
            GUIStyle fontStyle = new GUIStyle();
            fontStyle.fontSize = fontSize;
            fontStyle.normal.textColor = Color.white;
            string text = GameState.stateText;
            if (GUI.Button(new Rect(
                Screen.width / 2 - 50,
                Screen.height / 2 - 40,
                100,
                40),
                text,
                fontStyle))
            {
                if (GameState.state == GameState.GAMEOVER ||
                    GameState.state == GameState.GAMECOMPLETE)
                {
                    Debug.Log("Reload from start menu");
                    Scoring.lives = 3;
                    Scoring.gameScore = 0;
                    GameState.level = 1;
                    GameState.state = GameState.PLAYING;
                    SceneManager.LoadScene(0);
                }
            }
        }
    }
    //*/
}
