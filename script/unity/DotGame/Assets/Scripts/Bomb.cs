using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bomb : MonoBehaviour
{
    private bool inUse = false;
    private float fuseTimer;
    public float fuseLength = 1.0f;

    // Start is called before the first frame update
    void Start()
    {
        inUse = false;
        fuseTimer = fuseLength;
    }

    // Update is called once per frame
    void Update()
    {
        if (inUse)
        {
            fuseTimer -= Time.deltaTime;
            if (fuseTimer <= 0.0f)
            {
                gameObject.SetActive(false);
                // Destroy(gameObject);
            }
        }
    }

    public void Use()
    {
        this.inUse = true;
    }
}
