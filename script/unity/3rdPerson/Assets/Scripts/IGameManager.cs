using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IGameManager
{
    ManagerStatus status { get; }
    void Startup();
}

public enum ManagerStatus
{
    Shutdown,
    Initializing,
    Started
}