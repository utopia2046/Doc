using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Scoring : MonoBehaviour
{
    const int fontSize = 30;
    public static int gameScore = 0;

    private void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
    }

    void Update()
    {
    }

    private void OnGUI()
    {
        // Draw Score on Top Left corner
        GUI.skin.box.fontSize = fontSize;
        Rect scoreRect = new Rect(20, 20, 360, 50);
        string levelText = "Level: " + GameState.level.ToString();
        string scoreText = "Score: " + gameScore.ToString();
        GUI.Box(scoreRect, levelText + " " + scoreText);
        // Draw Game Over
        if (GameState.state == GameState.GAMEOVER || GameState.state == GameState.LEVELCOMPLETE)
        {
            GUIStyle fontStyle = new GUIStyle();
            fontStyle.fontSize = fontSize;
            fontStyle.normal.textColor = Color.white;
            string text = GameState.inBox ? "Success" : "Fail";
            //GUI.skin.box.fontSize = 36;
            //GUI.Box(new Rect(
            //    Screen.width / 2 - 200,
            //    Screen.height / 2 - 40,
            //    400,
            //    80),
            //GameState.inBox ? "Success" : "Fail");
            if (GUI.Button(new Rect(
                Screen.width / 2 - 50,
                Screen.height / 2 - 40,
                100,
                40),
                text,
                fontStyle))
            {
                GameState.state = GameState.PLAYING;
            }
        }
    }
}
