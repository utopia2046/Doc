using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameState : MonoBehaviour
{
    public static int state;
    public const int PLAYING = 1;
    public const int GAMEOVER = 2;
    public const int LEVELCOMPLETE = 3;
    public static bool inBox = false;

    private void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
        state = PLAYING;
        inBox = false;
    }
}
