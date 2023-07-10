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
