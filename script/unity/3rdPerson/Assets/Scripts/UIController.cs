using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIController : MonoBehaviour
{
    [SerializeField] SettingsPopup popup;

    // Start is called before the first frame update
    void Start()
    {
        popup.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyUp(KeyCode.M))
        {
            bool isShowing = popup.gameObject.activeSelf;
            popup.gameObject.SetActive(!isShowing); // toggle settings popup show/hide
        }
    }
}
