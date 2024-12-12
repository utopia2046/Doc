using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    public string FirstSceneName = "Level1";

    public void PlayGame()
    {
        Debug.Log("Loading scene: " + FirstSceneName);
        Scoring.gameScore = 0;
        Scoring.lives = 3;
        GameState.state = GameState.PLAYING;
        SceneManager.LoadScene(FirstSceneName); // or LoadScene(1), which uses scene index set in build settings
    }

    // Update is called once per frame
    public void QuitGame()
    {
        Debug.Log("Quitting game");
        Application.Quit();
    }
}
