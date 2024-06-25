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
2. Double click animation controller to open animator window and edit state changing conditions.

Suggested Poeject Structure:

- Assets\
  - Scenes\
  - Sprites\
    - Enemies\
    - Player\

<!-- TODO: unfinished below -->

## World Building

## Interactive Items

## Health and Inventory

## Characters, Coroutines, and Spawn Points

## Artificial Intelligence and Slingshots
