using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioManager : MonoBehaviour, IGameManager
{
    [SerializeField] AudioSource soundSource;
    [SerializeField] AudioSource music1Source;
    [SerializeField] AudioSource music2Source;
    [SerializeField] string introBGMusic;
    [SerializeField] string levelBGMusic;

    public ManagerStatus status { get; private set; }

    private float _musicVolume;
    public float musicVolume
    {
        get
        {
            return _musicVolume;
        }
        set
        {
            _musicVolume = value;
            if (music1Source != null)
            {
                music1Source.volume = _musicVolume;
            }
            if (music2Source != null)
            {
                music2Source.volume = _musicVolume;
            }
        }
    }

    public float soundVolume
    {
        get { return AudioListener.volume; }
        set { AudioListener.volume = value; }
    }

    public bool soundMute
    {
        get { return AudioListener.pause; }
        set { AudioListener.pause = value; }
    }

    public bool musicMute
    {
        get
        {
            return (music1Source != null) ? music1Source.mute : false;
        }
        set
        {
            if (music1Source != null)
            {
                music1Source.mute = value;
            }
            if (music2Source != null)
            {
                music2Source.mute = value;
            }
        }
    }

    public void Startup()
    {
        Debug.Log("Audio manager starting...");

        music1Source.ignoreListenerVolume = true;
        music1Source.ignoreListenerPause = true;
        music2Source.ignoreListenerVolume = true;
        music2Source.ignoreListenerPause = true;

        soundVolume = 1f;
        musicVolume = 1f;

        status = ManagerStatus.Started;
    }

    public void PlaySound(AudioClip clip)
    {
        soundSource.PlayOneShot(clip);
    }

    public void PlayMusic(AudioClip clip)
    {
        music1Source.clip = clip;
        music1Source.Play();
    }

    public void StopMusic()
    {
        music1Source.Stop();
    }

    public void PlayIntroMusic()
    {
        PlayMusic(Resources.Load($"Music/{introBGMusic}") as AudioClip);
    }

    public void PlayLevelMusic()
    {
        PlayMusic(Resources.Load($"Music/{levelBGMusic}") as AudioClip);
    }
}
