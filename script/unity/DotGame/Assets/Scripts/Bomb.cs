using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bomb : MonoBehaviour
{
    private bool inUse = false;
    private float fuseTimer;
    public ParticleSystem sparks;
    public ParticleSystem explosion;
    public GameObject explode;
    public float fuseLength = 1.0f;
    public float explosiveRadius = 1.5f;
    public AudioClip explodeSound;

    // Start is called before the first frame update
    void Start()
    {
        inUse = false;
        fuseTimer = fuseLength;
        //sparks = GetComponentInChildren<ParticleSystem>();
        //comparray = GetComponentsInChildren<ParticleSystem>();
        //foreach (ParticleSystem p in comparray)
        //{
        //    if (p.gameObject.name == "Explosion") explosion = p;
        //    if (p.gameObject.name == "Sparks") sparks = p;
        //}
        sparks.Stop();
        explosion.Stop();
    }

    // Update is called once per frame
    void Update()
    {
        if (inUse)
        {
            if (sparks.isStopped)
            {
                sparks.Play();
            }
            explode.SetActive(true);
            fuseTimer -= Time.deltaTime;
            if (fuseTimer <= 0.0f)
            {
                Debug.Log("Boom!");
                AudioSource.PlayClipAtPoint(explodeSound, Camera.main.transform.position);
                explosion.transform.SetParent(null);
                explosion.Play();
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
            if (obj.tag == "Enemy")
            {
                //Destroy(colliders[i].gameObject);
                colliders[i].gameObject.SetActive(false);
                if (obj.name == "Eye")
                {
                    Debug.Log("Kill an eye");
                    Scoring.gameScore += 50;
                }
                if (obj.name == "Robot")
                {
                    Debug.Log("Kill a robot");
                    Scoring.gameScore += 100;
                }
            }
        }
    }
}
