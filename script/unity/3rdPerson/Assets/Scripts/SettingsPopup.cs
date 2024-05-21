using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SettingsPopup : MonoBehaviour
{
    public void OnSoundToggle()
    {
        Debug.Log("Sound Toggled");
        Managers.Audio.soundMute = !Managers.Audio.soundMute;
    }

    public void OnSoundValue(float volume)
    {
        Debug.Log("Set volume = " + volume);
        Managers.Audio.soundVolume = volume;
    }
}
