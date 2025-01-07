using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class DottimaTitle : MonoBehaviour
{
    float timer = 3.0f;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        float delta;
        delta = Time.deltaTime;
        gameObject.transform.Translate(new Vector3(delta * 4.0f, 0.0f, 0.0f));
        timer -= delta;
        if (timer < 0)
        {
            SceneManager.LoadScene(0);
        }
    }
}
