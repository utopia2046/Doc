# Unity in Action: Multiplatform Game Development in C#, by Joseph Hocking, 3rd Edition (2022)

## Basics

In a component system, objects exist on a flat hierachy, and different objects have different collections of components.

![Inheritance vs. Composition](../images/Inheritance%20vs.%20Composition.png)

All scripts that inherit from `MonoBehaviour` (base class of script components) are components. `MonoBehaviour` defines run methods such as `Start()` (called once when the object is loaded), and `Update()` (called every frame). Scripts are attached to `GameObject`, and code is written inside scripts.

Hello World Project

1. Create a new project.
2. Create a new C# script.
3. Create an empty GameObject.
4. Drag the script onto the object.
5. Add the log command to the script.
6. Click Play!

## Create a simple 2D RPG

### 1. Create project

1. Create project using "2D Core" template.
2. Add [MiniFantasy Resources](https://assetstore.unity.com/?q=minifantasy&orderBy=1) to My Assets, click Open in Unity.
3. Window -> Package Manager -> Import MiniFantasy resources -> Select Sprites only -> Import.

### 2. Create map

1. In Hierachy window, right click -> 2D Object -> Tilemap -> Rectangular, rename it as "Ground".
2. In Scene window -> Open Tile Palette -> Create New Palette -> name it as "Plains" -> Create new folder under resources and name it as "Tiles".
3. In Project window, select "Forgotten Plains" -> Sprites -> Tileset -> Tiles.
4. In Inspector, make sure Filter Mode is "Point (no filter)", and Compression is "None", click Sprite Editor -> Slice -> Grid by Cell Size -> set as 8x8 -> Slice -> Apply.
5. Drag Tiles from Project window to Tile Palette window, select "Resources/Tiles" folder. Use mouse scroll to zoom in/out, alt + drag or right mouse drag to pan.
6. Click any tile or tiles, and draw it on Scene window. U for box fill and E for erase.
7. Create another Palette of plants under "Resources/Tiles".
8. Create another Tilemap named "Plant" and drag it under "Ground", draw grass and flowers on it.
9. Create Tilemap named "Obstacle" and draw trees and rocks on it.
10. In Inspector -> Add Component -> Tilemap -> Tilemap Collider 2D.

### 3. Create actor

1. Create a new folder under "Resources" and name it as "Prefabs:.
2. Under Hierachy, create 2D object -> Sprites -> Square, name it as "Player".
3. In Inspector -> Additional Settings -> Sorting Layer -> Add Sorting Layers -> + -> name as "Player" -> change the sprite's layer to be on it.
4. In Inspector -> Sprite -> input "HumanBase" -> select a sprite.
5. Under Resources folder, create a new folder named "Scripts". Create C# script in the folder named "PlayerMovement", sample code is as below.
6. Windows -> Package Manager -> Select Unity Registry -> Search for "Input System" -> Install.
7. In Inspector, Add Component -> Physics 2D -> Rigidbody 2D, set Gravity Scale to be 0.
8. In Inspector, Add Component -> Input -> Player Input -> Create Actions -> name it as "PlayerInputActions" and use default settings.
9. In Inspector, Add Component -> Script -> Select "PlayerMovement", this link the movement script to Player sprite.

- Set sprite to be "multiple" to use a series of sprites as animation.
- Set Flip X to flip the character sprite on moving left & right.
- Use Packagae "cenamatic" for camera zooming and moving.
- Create a character and drag it to "prefab", create this character as a class, later you can create multiple instances of this character.
- Add combat animation to player sprite.
- Add collider for the weapon so it could hit the enemy sprite.
- Create an interface call IDamageable and declare OnHit method in it. Inherit and implement this interface in class DamageableCharacter, then add this class as component for player character and enemies (notice the component system difference with traditional OO inheritance).
- Add multiple animations like PlayerWalk, PlayerIdle, PlayerHit, PlayerDamage, PlayerDie, OrcHit, OrcDamage, OrcDie, etc. Specify state transition in Animator window and transition condition in Inspector window.
- In DamageableCharacter, add a method to destroy the character when it dies. In animation window, add event at last frame of orc die animation.
- Add 3D object -> Text mesh-pro and TMP essentials to draw text on screen, name it as DamagePopup and drag it to Prefabs.

``` csharp
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerMovement : MonoBehaviour
{
    Rigidbody2D rb;
    Vector2 moveInput;
    public float moveSpeed;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    private void OnMove(InputValue value)
    {
        moveInput = value.Get<Vector2>();
    }

    private void FixedUpdate()
    {
        rb.AddForce(moveInput * moveSpeed);
    }
}
```

## Basic First-Person Shooter (FPS)

1. Set up the room: create the floor, outer walls, and inner walls.
2. Place the lights and camera.
3. Create the player object (including attaching the camera on top).
4. Write movement scripts: rotate with the mouse and move with the keyboard.

| Notice that Unity uses a left-handed coordinate system, as do many 3D art applications.

<!--
TODO: unfinished below here

## 3D camera control

## Raycasting

## 2D graphics

## 2D physics

## Game GUI

## Manage inventory

## Interactive devices and items in game

## Sound effects and music

## Deploy to desktop, web, or mobile

-->
