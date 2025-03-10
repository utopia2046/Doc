using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class GameMenu : MonoBehaviour
{
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(0, 0, Screen.width, Screen.height));
        GUILayout.BeginVertical();
        GUILayout.FlexibleSpace();
        if (GUILayout.Button("Back", GUILayout.Width(200.0f)))
        {
            //Application.LoadLevel("Menu");
            SceneManager.LoadScene("Menu");
        }
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}
