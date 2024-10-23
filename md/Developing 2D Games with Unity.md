# Developing 2D Games with Unity, Independent Game Programming with C#, by Jared Halpern

Source code and Assets: <https://github.com/Apress/Devel-2D-Games-Unity>

Resources:

- <https://gamedev.stackexchange.com>
- <https://forum.unity.com>
- <https://answers.unity.com>
- <https://globalgamejam.org>
- <https://freesound.org>
- <https://bigsoundbank.com/>
- <https://pixabay.com/>

External Tools:

- Graphics creation: Blender, GIMP
- Sound Effects: Audacity
- Music: MuseScore

## Foundation

Sprite Tool - Pixel Dude Maker: <https://0x72.itch.io/pixeldudesmaker>

Import Sprite Sheet:

1. Add Sprite Renderer Component to Player GameObject
2. Import Sprite Sheet image, set `Texture Type` as `Sprite (2D and UI)`, `Sprite Mode` as `Multiple`
3. Use Sprite Editor to slice up the sprite sheet into individual sprites
4. Set Filter Mode to "Point (no filter)" and Compression to None

Add animation:

1. Select frames in imported sprite sheet, drag to GameObject to create animation clips and a controller.
2. Double click animation controller to open animator window, create animator parameters for states and edit state changing conditions.
3. In code, control state change by changing the value of animator parameters.

``` csharp
// To change animator parameter value in code
Animator animator;
// in Start()
animator = GetComponent<Animator>();
// in state change logic
animator.SetInteger(parameterName, value);
```

Suggested Poeject Structure:

- Assets\
  - Scenes\
  - Sprites\
    - Enemies\
    - Player\

Tips:

1. When the 2D game is top down view, remember to turn off gravity (Edit -> Project Settings -> Physics 2D -> Gravity).
2. Layers are used in collision detection to determine which layers are aware of each other and thus can interact.
3. Sorting Layers (in Renderer settings) are used by Unity rendering engine to describe the order in which to render pixels.

<!-- TODO: unfinished below -->

## World Building

Create Tilemap

1. Import tilemap resource image, then slice it into SpriteSheet.
2. Create a tilemap GameObject: Create -> 2D Object -> Tilemap -> Rectangle (or hexagonal, depends on your game).
3. Create Tile Palette: Window -> 2D -> Tile Palette -> Create New Palette.
4. Drag spritesheet into tile palette, then use brush tool to paint selected tile on tilemap (use `[]` keys to rotate the tile).
5. To paint multiple tile on same grid (object on background), we need multiple tilemaps and stack them in Sorting Layers (notice that the Sorting Layer setting is under Tilemap Renderer, not on top right of Inspector).

Camera Size and PPU (Pixels per Unit): `(Vertical resolution / PPU) * 0.5 = Camera Size`

Cinamachine Virtual Camera

1. Install Cinemachine using Package Manager.
2. Create Object -> Cinemachine -> 2D Camera, which creates a virtual camera object and add CinemachineBrain component to Main Camera.
3. Drag Player object on CM vcam `Follow`, the vcam will follow player movement. To set `Follow` property programmatically, use code like `virtualCamera.Follow = player.transform;`.
4. Setting `Dead Zone Width` and Height on CM vcam, the map won't scroll until player reach dead zone border.
5. To set map border, on CM vcam, add Extension -> Cinemachine Confiner -> Bounding Sape 2D -> add a Polygon Collider to Layer_Ground and attach it here.

| Tips: To create a pixel-perfect sprite, use `Material` with shader as `Spites/Default`, and make sure that `Pixel Snap` is checked.

Use Tilemap Collider to limit Player's walking area

1. Use a tilemap for impenatrable objects and set it on a separate sorting layer, add `Tilemap Collider 2D` component on this tilemap layer, all object rendered on this tilemap will have colliders around them;

## Interactive Items

To make a GameObject (item) interactive:

1. Add a collider on it so that it could be touched;
2. Check collider setting `Is Trigger` (!important);
3. Add a Tag to the object so that in script we could detect it;

### Layer-based Collision Detection

1. Create Layer for consumables items and enemies (who will not consume items).
2. In Edit menu -> Project Settings -> Physics 2D, set Layer Collision Matrix, uncheck consumables & enemies layer checkbox so that enemies won't be aware of consumables.
3. Implement `OnTriggerEnter2D` method in player script for interacting with consumable items, sample like below.

``` csharp
void OnTriggerEnter2D(Collider2D collision)
{
    if (collision.gameObject.CompareTag("CanBePickedUp")) // CanBePickedUp is the tag we add on consumable items
    {
        Debug.Log("Item collected");
        // ... logic that manage the item in inventory
        collision.gameObject.SetActive(false); // hide the collected item
    }
}
```

### Scriptable Objects

- Used to store data
- Defined once and multiple references to save memory
- Scriptable Object instances are stored in project as separate asset files, their properties can be modified in Inspector
- Inherit from `ScriptableObject` class instead of `MonoBebaviour`, can't be attached to `GameObjects`
- Create reference from inside Unity scripts that inherit from `MonoBehaviour`

For example:

1. Create a `HitPoints` Scriptable Objects class with a public member called `value`, then create an instance from it also called `HitPoints`;
2. Add `HitPoints hitPoints` property in both `Player` and `HealthBar` GameObject, then attach both properties to the same `HitPoint` instance, the 2 objects will share the same hitPoints value.

Also, we could create `Consumable` GameObject with a `Item` Scriptable Objects instance slot, the `Item` type and other properties like stackable could be shared between collectable coin, heart, player (on collision), and inventory game objects.

## Health and Inventory

The order in which objects appear in the hierarchy view is the order in which they’ll be rendered. the top-most objects in the hierarchy will be rendered first and the bottom last, resulting in the top-most objects appearing in the background.

### Masked Image

1. Create a UI->Image object and add component `Mask` on it, any child object of it will be masked automatically;
2. Create a child Image of previous image with mask component, select its source image, and set `Image Type` to be `Filled`, and choose `Fill Method` and `Fill Origin` depends on how you want the mask uncovers. By changing `Fill Amount`, we could see how the mask works.
3. In script, programmatically update the `FillAmount` property to update the masked image, for example, a health bar.

### Customized Fonts

1. Create a `Fonts` folder under `Assets` and drag .ttf file in it.
2. Right click the imported font, `Create` -> `TextMeshPro` -> `Font Asset`.
3. Select the created font asset in TMP text's Inspector window, or set it as default in `Project Setting` -> `TextMeshPro` -> `Settings` -> `Default Font Asset`.

| Tips: `Horizontal Layout Group` component will automatically arrange for all its subviews to be placed alongside each other horizontally.

| Tips: Fix TMP Text not shown, at right top of the Material, `Create Material Preset` using shader `TextMeshPro/Distance Field`, make sure `Color` -> `Alpha` is not 0, then assign it to the text's `Material Preset`.

## Characters, Coroutines, and Spawn Points

Instead of using `Update` method in `MonoBehaviour` class, we could also use `InvokeRepeating` method to repeat regularly.

``` csharp
// in Start()
if (repeatInterval > 0)
{
    // InvokeRepeating(methodToCall, timeToWaitBeforeFirstCall, timeIntervalToWaitBetweenInvocations)
    InvokeRepeating("SpawnObject", 0.0f, repeatInterval);
}
```

We could use `Instantiate` method to create new instance of a prefab

``` csharp
if (prefabToSpawn != null)
{
    // Instantiate(prefab, position, rotation)
    return Instantiate(prefabToSpawn, transform.position, Quaternion.identity);
    // transform.position is the Vector3 of current GameObject
    // Quaternion.identity means no rotation
}
```

### Get Certain GameObject in Script

1. Create a public property in `MonoBehaviour` script, and drag the target GameObject instance in Inspector.
2. To visit the GameObject current script attached to, use `gameObject` in `MonoBehaviour`.
3. Use static `GameObject.FindWithTag` method.
4. To get a GameObject that we hit using Collider. use built-in method `OnCollisionEnter2D(Collision2D collision)`, and call `collision.gameObject.CompareTag("TargetTag")` to get target GameObject with certain tag.

``` csharp
void OnCollisionEnter2D(Collision2D collision)
{
    if (collision.gameObject.CompareTag("Player"))
    {
        // ...
    }
}
```

To get a component that attached to a certain GameObject, use `gameObject.GetComponent<{ComponentType}>`.

Unity will force a method to be called within single frame, for long ngrunning methods that are intended to execute over the course of multiple frames are often implemented as **Coroutines**.

``` csharp
// Define a Coroutine
public IEnumerator RunEveryThreeSeconds()
{
    while (true)
    {
        print("I will print every three seconds.");
        yield return new WaitForSeconds(3.0f);
    }
}

// call the Coroutine
StartCoroutine(RunEveryThreeSeconds());
```

## Artificial Intelligence and Slingshots

To measure distance of 2 points, use `Vector3.sqrMagnitude` method like:

``` csharp
float remainingDistance = (transform.position - endPosition).sqrMagnitude;
```

To move a GameObject, we need to have RigidBody2D component attached on it, and move the GameObject like:

``` csharp
// Vector3.MoveTowards(currentPosition, targetPosition, distanceToMoveInFrame)
Vector3 newPosition = Vector3.MoveTowards(rigidBodyToMove.position, endPosition, speed * Time.deltaTime);
rigidBodyToMove.MovePosition(newPosition);
```

### Object Pooling

Instantiating and Destroying objects `Destroy(gameObject)` in Unity is more performance intensive than simply activating and deactivating `gameObject.SetActive(false)`. We can use a technique called **Object Pooling** to maintain good performance.

1. Pre-instantiate multiple copies of an object for the scene ahead of time, de-activate them, and add them to an object pool.
2. When the scene requires an object, loop through the object pool, return the first inactive object found, and activate it.
3. When the scene is finished using the object, place it inactive, and return it to the object pool to be re-used by the scene in the future.

``` csharp
// check gameObject active
if (gameObject.activeSelf == false)
{
    // set gameObject active
    gameObject.SetActive(true);
}
```

| Tips: `Vector3.Lerp` calculates Linear Interpolation between start point and destination. It is useful for calculating object moving path.

``` csharp
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
```

### Blend Trees

Blend Trees can be used to smoothly blend multiple animations into one smooth animation, such as walking and gradually begins to run, or firing an arm during running. A Blend Tree is controlled by variables that are configured in the Unity Editor and set in code.

#### Create a Blend Tree

1. Right-click in the Animator window and select: `Create State` -> `from New Blend Tree`.
2. Select the created Blend Node and change its name in the Inspector to: "Walk Tree".
3. Double-click the Walk Tree node to view the Blend Tree Graph.
4. Select the Blend Tree node and change the `Blend Type` in the Inspector to: `2D Simple Diectional`.
5. Select the Blend Tree node, right-click, and select: `Add Motion`.
6. In the Inspector, click the dot next to the Motion we just added to open the `Select Motion` selector.
7. With the Select Motion selector open, select the player-walk-east animation clip.
8. Add three more motions and add the following animation clips: player-walk-south, player-walkwest, and player-walk-north.
9. Click the `Base Layer` button to go back to the base Animator view.
10. Create these three Animation Parameters:
    - isWalking of type: Bool
    - xDir of type: Float
    - yDir of type: Float
11. With the Blend Tree selected, select the xDir and yDir parameters from the dropdown in the Inspector.
12. Set the X and Y positions for the motions accordingly. For example, the player-walk-south motion positions should be set to (0, -1)
13. In script, set animator parameters to affect the animation state machine.

    ``` csharp
    // 1
    movement.x = Input.GetAxisRaw("Horizontal");
    movement.y = Input.GetAxisRaw("Vertical");
    // 2
    animator.SetBool("isWalking", true);
    // 3
    animator.SetFloat("xDir", movement.x);
    animator.SetFloat("yDir", movement.y);
    ```

14. Select each one of the four child nodes of the blend tree and if it is not checked by default, check the `Loop Time` property.
15. Right-click on the Idle State node in the Animator and select: `Make Transition`. Connect the transition to the Walking Blend Tree. Select the transition and use the following settings:
    - Has Exit Time: unchecked
    - Fixed Duration: unchecked
    - Transition Duration: 0
16. Create a Condition using the `isWalking` variable we created. Set it to: `true`.
17. Create another transition between the Walking Blend Tree and the Idle state. Select the transition and use the similar settings as earlier.

### Bouncing with built-in component

1. Add `Rigid Body 2D` and `Circle Collider 2D` (or other colliders) to the moving object (e.g. a ball).
2. In Project, click Add -> 2D -> Physics Material 2D, then name it as `Bounce`, and change the `Bounciness` property to a value between 0 (no bounce) and 1 (full bounce).
3. Drag the `Bounce` material into the ball object's `Circle Collider 2D` -> `Material` property.
4. Add `Box Collider 2D` to the ground object (or wall or any other obstacles).

Make sure you have `Gravity` in `Project Settings` -> `Physics 2D`. When the moving object hits the obstacle, it will now bounce back.

### Preserve Object between Scenes

You can load a new scene using `SceneManager`, but to preserve objects, such as game state between scenes, use `DontDestroyOnLoad` on the object to be reserved.

``` csharp
// load new scene
using UnityEngine.SceneManagement;
SceneManager.LoadScene("Scenes/Level 2");

// on the obejct to be reserved
private void Awake()
{
    DontDestroyOnLoad(gameObject);
}
```

