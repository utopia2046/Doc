using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BasicUI : MonoBehaviour
{
    int posX = 10;
    int posY = 10;
    int width = 100;
    int height = 30;
    int buffer = 10;
    int fontSize = 18;
    private GUIStyle currentStyle = null;

    void OnGUI()
    {
        List<string> itemList = Managers.Inventory.GetItemList();
        //InitStyles(fontSize);

        if (itemList.Count == 0)
        {
            GUI.Box(new Rect(posX, posY, width, height), "No Items"); //, currentStyle
            return;
        }

        foreach (string item in itemList)
        {
            int count = Managers.Inventory.GetItemCount(item);
            Texture2D image = Resources.Load<Texture2D>($"Icons/{item}");
            GUI.Box(new Rect(posX, posY, width, height), new GUIContent($"({count})", image)); //, currentStyle
            posX += width + buffer;
        }
    }

    private void InitStyles(int fontSize)
    {
        if (currentStyle == null)
        {
            //Texture2D background = new Texture2D(1, 1, TextureFormat.RGBAFloat, false);
            //background.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.5f));
            //background.Apply();

            currentStyle = new GUIStyle(); // new GUIStyle(GUI.skin.box)
            //currentStyle.normal.background = MakeTex(2, 2, new Color(0f, 0f, 0f, 0.5f)); //background;
            currentStyle.normal.textColor = Color.white;
            currentStyle.fontSize = fontSize;
            currentStyle.fontStyle = FontStyle.Bold;
        }
    }

    private Texture2D MakeTex(int width, int height, Color col)
    {
        Color[] pix = new Color[width * height];
        for (int i = 0; i < pix.Length; ++i)
        {
            pix[i] = col;
        }
        Texture2D result = new Texture2D(width, height);
        result.SetPixels(pix);
        result.Apply();
        return result;
    }
}
