using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TitleMenu : MonoBehaviour
{
    public string GameScene;

    public void PlayGame()
    {
        SceneManager.LoadScene(GameScene);
    }

    public void ExitGame()
    {
        Application.Quit();
    }
}