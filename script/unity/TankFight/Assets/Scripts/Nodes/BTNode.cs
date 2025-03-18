using UnityEngine;
using System.Collections;

[System.Serializable]
public abstract class BTNode
{

    /* Delegate that returns the state of the node.*/
    public delegate NodeStates NodeReturn();

    /* The current state of the node */
    protected NodeStates m_nodeState;

    public NodeStates nodeState
    {
        get { return m_nodeState; }
    }

    /* The constructor for the node */
    public BTNode() { }

    /* Implementing classes use this method to valuate the desired set of conditions */
    public abstract NodeStates Evaluate();

}
