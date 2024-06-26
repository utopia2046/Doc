using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BasicUI : MonoBehaviour
{
    [SerializeField] int fontSize = 18;
    private GUIStyle currentStyle = null;

    void OnGUI()
    {
        int posX = 10;
        int posY = 10;
        int width = 100;
        int height = 30;
        int buffer = 10;

        List<string> itemList = Managers.Inventory.GetItemList();

        if (currentStyle == null)
        {
            currentStyle = MakeStyle(fontSize);
        }

        if (itemList.Count == 0)
        {
            GUI.Box(new Rect(posX, posY, width, height), "No Items", currentStyle);
            return;
        }

        // show item list
        foreach (string item in itemList)
        {
            int count = Managers.Inventory.GetItemCount(item);
            Texture2D image = Resources.Load<Texture2D>($"Icons/{item}");
            GUI.Box(new Rect(posX, posY, width, height), new GUIContent($"({count})", image), currentStyle);
            posX += width + buffer;
        }

        // show equipped item
        string equipped = Managers.Inventory.equippedItem;
        if (equipped != null)
        {
            posX = Screen.width - (width + buffer);
            Texture2D image = Resources.Load($"Icons/{equipped}") as Texture2D;
            GUI.Box(new Rect(posX, posY, width, height), new GUIContent("Equipped", image));
        }

        posX = 10;
        posY += height + buffer;

        // equip button
        foreach (string item in itemList)
        {
            if (GUI.Button(new Rect(posX, posY, width, height), $"Equip {item}"))
            {
                Managers.Inventory.EquipItem(item);
            }

            if (item == "health")
            {
                if (GUI.Button(new Rect(posX, posY + height + buffer, width, height), "Use Health"))
                {
                    Managers.Inventory.ConsumeItem("health");
                    Managers.Player.ChangeHealth(25);
                }
            }

            posX += width + buffer;
        }
    }

    private GUIStyle MakeStyle(int fontSize)
    {
        GUIStyle style = new GUIStyle(GUI.skin.box);
        style.normal.background = MakeTexture(new Color(0f, 0f, 0f, 0.5f));
        style.normal.textColor = Color.white;
        style.fontSize = fontSize;
        style.fontStyle = FontStyle.Bold;

        return style;
    }

    private Texture2D MakeTexture(Color backgroundColor)
    {
        Texture2D background = new Texture2D(1, 1, TextureFormat.RGBAFloat, false);
        background.SetPixel(0, 0, backgroundColor);
        background.Apply();
        return background;
    }
}
