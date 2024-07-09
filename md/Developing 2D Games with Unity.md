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

Install Cinemachine using Package Manager, then we can use virtual cameras using this package

## Interactive Items

## Health and Inventory

## Characters, Coroutines, and Spawn Points

## Artificial Intelligence and Slingshots
