using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameState : MonoBehaviour
{
    public static int state;
    public const int gamePlay = 1;
    public const int gameOver = 2;
    public static bool inBox = false;

    void Start()
    {
        state = gamePlay;
    }
}
