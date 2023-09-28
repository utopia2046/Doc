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
