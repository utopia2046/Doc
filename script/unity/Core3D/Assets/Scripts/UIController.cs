using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class UIController : MonoBehaviour
{
    [SerializeField] TMP_Text scoreLabel;
    [SerializeField] SettingsPopup settingsPopup;

    // Start is called before the first frame update
    void Start()
    {
        settingsPopup.Close();
    }

    // Update is called once per frame
    void Update()
    {
        scoreLabel.text = Time.realtimeSinceStartup.ToString();
    }

    public void OnOpenSettings()
    {
        Debug.Log("open settings");
        settingsPopup.Open();
    }
    
    public void OnCloseSettings()
    {
        Debug.Log("close settings");
        settingsPopup.Close();
    }

    public void OnMouseDown()
    {
        Debug.Log("pointer down");
    }
}
