using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SettingsMenu : MonoBehaviour
{
    public TMP_Dropdown resDropdown;
    Resolution[] resolutions;

    void Start()
    {
        resolutions = Screen.resolutions; // get all available resolutions
        resDropdown.ClearOptions();

        List<string> options = new List<string>();

        int currentIndex = 0;
        Debug.Log("resolutions list length = " + resolutions.Length);
        for (int i = 0; i < resolutions.Length; i++)
        {
            string option = resolutions[i].width + " x " + resolutions[i].height;
            options.Add(option); // add options

            // get current screen resolution as current option
            if (resolutions[i].width == Screen.currentResolution.width &&
                resolutions[i].height == Screen.currentResolution.height)
            {
                Debug.Log("Current resolution: " + Screen.currentResolution.width +
                    " x " + Screen.currentResolution.height);
                currentIndex = i;
            }
        }

        resDropdown.AddOptions(options);
        resDropdown.value = currentIndex;
        resDropdown.RefreshShownValue();
    }

    public void SetResolution(int resolutionIndex)
    {
        Resolution res = resolutions[resolutionIndex];
        Screen.SetResolution(res.width, res.height, Screen.fullScreen); // set resolution
    }

    public void SetFullScreen(bool isFullScreen)
    {
        Screen.fullScreen = isFullScreen; // toggle full screen mode
    }
}
