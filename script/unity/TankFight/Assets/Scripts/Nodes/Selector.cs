using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Selector : BTNode
{
    /** The child nodes for this selector */
    protected List<BTNode> m_nodes = new List<BTNode>();


    /** The constructor requires a lsit of child nodes to be
     * passed in*/
    public Selector(List<BTNode> nodes)
    {
        m_nodes = nodes;
    }

    /* If any of the children reports a success, the selector will
     * immediately report a success upwards. If all children fail,
     * it will report a failure instead.*/
    public override NodeStates Evaluate()
    {
        foreach (BTNode node in m_nodes)
        {
            switch (node.Evaluate())
            {
                case NodeStates.FAILURE:
                    continue;
                case NodeStates.SUCCESS:
                    m_nodeState = NodeStates.SUCCESS;
                    return m_nodeState;
                case NodeStates.RUNNING:
                    m_nodeState = NodeStates.RUNNING;
                    return m_nodeState;
                default:
                    continue;
            }
        }
        m_nodeState = NodeStates.FAILURE;
        return m_nodeState;
    }
}
