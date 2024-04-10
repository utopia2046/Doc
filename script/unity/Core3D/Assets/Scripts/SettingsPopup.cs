using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SettingsPopup : MonoBehaviour
{
    [SerializeField] Slider speedSlider;
    [SerializeField] InputField nameInput;

    // Start is called before the first frame update
    void Start()
    {
        speedSlider.value = PlayerPrefs.GetFloat("speed", 1);
        nameInput.text = PlayerPrefs.GetString("name");
    }

    public void Open()
    {
        gameObject.SetActive(true);
    }

    public void Close()
    {
        gameObject.SetActive(false);
    }

    public void OnSubmitName(string name)
    {
        PlayerPrefs.SetString("name", name);
        Debug.Log($"Name: {name}");
    }

    public void OnSpeedValue(float speed)
    {
        PlayerPrefs.SetFloat("speed", speed);
        Debug.Log($"Speed: {speed}");
    }
}
