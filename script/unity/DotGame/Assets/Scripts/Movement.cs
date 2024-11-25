using UnityEngine;

public enum Directions
{
    Up,
    Right,
    Down,
    Left
}

public class Movement
{
    public static Vector2 GetDirectionVector(Directions direction)
    {
        switch (direction)
        {
            case Directions.Up:
                return new Vector2(0.0f, 1.0f);
            case Directions.Down:
                return new Vector2(0.0f, -1.0f);
            case Directions.Left:
                return new Vector2(-1.0f, 0.0f);
            case Directions.Right:
                return new Vector2(1.0f, 0.0f);
        }

        return new Vector2(0.0f, 1.0f);
    }

    public static Vector2 RetreatALittle(Vector2 currentPosition, Vector2 dirVector)
    {
        Vector2 newPosition = new Vector2(
            currentPosition.x - dirVector.x * 0.1f,
            currentPosition.y - dirVector.y * 0.1f);
        return newPosition;
    }
}
