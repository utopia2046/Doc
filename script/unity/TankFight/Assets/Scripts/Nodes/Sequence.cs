﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Sequence : BTNode
{
    /** Chiildren nodes that belong to this sequence */
    private List<BTNode> m_nodes = new List<BTNode>();

    /** Must provide an initial set of children nodes to work */
    public Sequence(List<BTNode> nodes)
    {
        m_nodes = nodes;
    }

    /* If any child node returns a failure, the entire node fails. Whence all
     * nodes return a success, the node reports a success. */
    public override NodeStates Evaluate()
    {
        bool anyChildRunning = false;

        foreach (BTNode node in m_nodes)
        {
            switch (node.Evaluate())
            {
                case NodeStates.FAILURE:
                    m_nodeState = NodeStates.FAILURE;
                    return m_nodeState;
                case NodeStates.SUCCESS:
                    continue;
                case NodeStates.RUNNING:
                    anyChildRunning = true;
                    continue;
                default:
                    m_nodeState = NodeStates.SUCCESS;
                    return m_nodeState;
            }
        }
        m_nodeState = anyChildRunning ? NodeStates.RUNNING : NodeStates.SUCCESS;
        return m_nodeState;
    }
}
