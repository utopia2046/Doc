using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class TankSceneManager : MonoBehaviour
{

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(Screen.width * 0.3f, Screen.height * 0.3f, Screen.width - (Screen.width * 0.6f), Screen.height - (Screen.height * 0.3f)));
        if (GUILayout.Button("Simple Navigation"))
        {
            //Application.LoadLevel("NavMesh01-Simple");
            SceneManager.LoadScene("NavMesh01-Simple");
        }
        else if (GUILayout.Button("Simple Navigation with Slopes"))
        {
            //Application.LoadLevel("NavMesh02-Slope");
            SceneManager.LoadScene("NavMesh02-Slope");
        }
        else if (GUILayout.Button("Navigation Mesh with Layers"))
        {
            //Application.LoadLevel("NavMesh03-Layers");
            SceneManager.LoadScene("NavMesh03-Layers");
        }
        else if (GUILayout.Button("Off Mesh Links"))
        {
            //Application.LoadLevel("NavMesh04-OffMeshLinks");
            SceneManager.LoadScene("NavMesh04-OffMeshLinks");
        }
        GUILayout.EndArea();

    }
}
