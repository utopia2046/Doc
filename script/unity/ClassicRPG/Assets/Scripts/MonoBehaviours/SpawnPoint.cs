using UnityEngine;

public class SpawnPoint : MonoBehaviour
{
    public GameObject prefabToSpawn;
    public float repeatInterval;
    public int maxInstances;
    int instanceNumber = 0;

    public void Start()
    {
        instanceNumber = 0;
        if (repeatInterval > 0)
        {
            InvokeRepeating("SpawnObject", 0.0f, repeatInterval);
        }
    }

    public GameObject SpawnObject()
    {
        if ((prefabToSpawn != null) && (instanceNumber < maxInstances))
        {
            instanceNumber++;
            // Instantiate(prefab, position, rotation)
            return Instantiate(prefabToSpawn, transform.position, Quaternion.identity);
        }

        return null;
    }
}
