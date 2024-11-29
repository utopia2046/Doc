using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bomb : MonoBehaviour
{
    private bool inUse = false;
    private float fuseTimer;
    public GameObject explode;
    public float fuseLength = 1.0f;
    public float explosiveRadius = 3.0f;

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
            explode.SetActive(true);
            fuseTimer -= Time.deltaTime;
            if (fuseTimer <= 0.0f)
            {
                Debug.Log("Boom!");
                DamageNearbyObjects(gameObject.transform);
                // Destroy(gameObject);
                gameObject.SetActive(false);
                explode.SetActive(false);
            }
        }
    }

    public void Use()
    {
        this.inUse = true;
    }

    void DamageNearbyObjects(Transform tr)
    {
        // find all colliders in 3.0f radius
        Collider2D[] colliders = Physics2D.OverlapCircleAll(tr.position, explosiveRadius);
        for (int i = 0; i < colliders.Length; i++)
        {
            GameObject obj = colliders[i].gameObject;
            if ((obj.tag == "Eye") || (obj.tag == "Robot") || (obj.tag == "Enemy"))
            {
                //Destroy(colliders[i].gameObject);
                colliders[i].gameObject.SetActive(false);
            }
        }
    }
}
