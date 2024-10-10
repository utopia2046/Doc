using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class NextLevel : MonoBehaviour
{
    public string NextLevelName = "";

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (GameState.state == GameState.LEVELCOMPLETE && GameState.inBox)
        {
            // bring next level button upfront
            Vector3 pos = transform.position;
            pos.z = 0;
            transform.position = pos;
        }
    }

    void OnMouseDown()
    {
        Debug.Log("Load next level name = " + NextLevelName + "; GameState.level = " + GameState.level);
        SceneManager.LoadScene(NextLevelName);
        GameState.level++;
    }
}
