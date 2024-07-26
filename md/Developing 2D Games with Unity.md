# Developing 2D Games with Unity, Independent Game Programming with C#, by Jared Halpern

Source code and Assets: <https://github.com/Apress/Devel-2D-Games-Unity>

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
3. Drag Player object on CM vcam `Follow`, the vcam will follow player movement.
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

## Health and Inventory

### Masked Image

1. Create a UI->Image object and add component `Mask` on it, any child object of it will be masked automatically;
2. Create a child Image of previous image with mask component, select its source image, and set `Image Type` to be `Filled`, and choose `Fill Method` and `Fill Origin` depends on how you want the mask uncovers. By changing `Fill Amount`, we could see how the mask works.
3. In script, programmatically update the `FillAmount` property to update the masked image, for example, a health bar.

### Customized Fonts

1. Create a `Fonts` folder under `Assets` and drag .ttf file in it.
2. Right click the imported font, `Create` -> `TextMeshPro` -> `Font Asset`.
3. Select the created font asset in TMP text's Inspector window, or set it as default in `Project Setting` -> `TextMeshPro` -> `Settings` -> `Default Font Asset`.

The order in which objects appear in the hierarchy view is the order in which they’ll be rendered. the top-most objects in the hierarchy will be rendered first and the bottom last, resulting in the top-most objects appearing in the background.

## Characters, Coroutines, and Spawn Points

## Artificial Intelligence and Slingshots
