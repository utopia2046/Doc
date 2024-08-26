using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Animator))]
public class Weapon : MonoBehaviour
{
    public GameObject ammoPrefab;
    static List<GameObject> ammoPool;
    public int poolSize;

    public float weaponVelocity;
    bool isFiring;

    void Awake()
    {
        // object pool
        if (ammoPool == null)
        {
            ammoPool = new List<GameObject>();
        }

        for (int i = 0; i < poolSize; i++)
        {
            GameObject ammoObject = Instantiate(ammoPrefab);
            ammoObject.SetActive(false);
            ammoPool.Add(ammoObject);
        }
    }

    void Update()
    {
        // check in every frame if user has clicked
        if (Input.GetMouseButtonDown(0))
        {
            FireAmmo();
        }
    }

    GameObject SpawnAmmo(Vector3 location)
    {
        foreach (GameObject ammo in ammoPool)
        {
            if (ammo.activeSelf == false)
            {
                ammo.SetActive(true);
                ammo.transform.position = location;
                return ammo;
            }
        }

        return null; // no inactive instance in pool, return null
    }

    void FireAmmo()
    {
        // Mouse uses Screen Space (Camera visible scope), we need to convert it to World Space (entire map in scene)
        Vector3 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        GameObject ammo = SpawnAmmo(transform.position);

        if (ammo != null)
        {
            Arc arcScript = ammo.GetComponent<Arc>();
            float travelDuration = 1.0f / weaponVelocity;
            StartCoroutine(arcScript.TravelArc(mousePosition, travelDuration));
        }
    }

    void OnDestroy()
    {
        ammoPool = null;
    }
}
