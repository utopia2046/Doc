using System.Collections;
using UnityEngine;

public class Arc : MonoBehaviour
{
    // Coroutine to move ammulation
    public IEnumerator TravelArc(Vector3 destination, float duration)
    {
        var startPosition = transform.position;
        var percentComplete = 0.0f;

        while (percentComplete <= 1.0f)
        {
            // Time.deltaTime is the time elapsed since the last frame was drawn
            percentComplete += Time.deltaTime / duration;
            var currentHeight = Mathf.Sin(Mathf.PI * percentComplete);
            // Vector3.Lerp calculates Linear Interpolation between start point and destination
            transform.position = Vector3.Lerp(startPosition, destination, percentComplete) + Vector3.up * currentHeight;

            yield return null;
        }

        // if arc has completed, deactivate
        gameObject.SetActive(false);
    }
}
