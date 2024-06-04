using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class UIController : MonoBehaviour
{
    [SerializeField] TMP_Text healthLabel;
    [SerializeField] InventoryPopup popup;

    void OnEnable()
    {
        Messenger.AddListener(GameEvent.HEALTH_UPDATED, OnHealthUpdated);
    }

    void OnDisable()
    {
        Messenger.RemoveListener(GameEvent.HEALTH_UPDATED, OnHealthUpdated);
    }

    void Start()
    {
        OnHealthUpdated();
        popup.gameObject.SetActive(false);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyUp(KeyCode.M))
        {
            bool isShowing = popup.gameObject.activeSelf;
            popup.gameObject.SetActive(!isShowing); // toggle settings popup show/hide
            popup.Refresh();
        }
    }

    public bool IsSettingsShown()
    {
        return popup.gameObject.activeSelf;
    }

    private void OnHealthUpdated()
    {
        string message = $"Health: {Managers.Player.health}/{Managers.Player.maxHealth}";
        healthLabel.text = message;
    }
}
