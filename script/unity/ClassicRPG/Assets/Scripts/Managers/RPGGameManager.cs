using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RPGGameManager : MonoBehaviour
{
    public static RPGGameManager sharedInstance = null;

    void Awake()
    {
        if (sharedInstance != null && sharedInstance != this) // only allow one instance
        {
            Destroy(gameObject);
        }
        else
        {
            sharedInstance = this;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        SetupScene();
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void SetupScene()
    {

    }
}
