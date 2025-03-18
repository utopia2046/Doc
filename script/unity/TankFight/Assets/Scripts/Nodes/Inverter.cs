using UnityEngine;
using System.Collections;

public class Inverter : BTNode
{
    /* Child node to evaluate */
    private BTNode m_node;

    public BTNode node
    {
        get { return m_node; }
    }

    /* The constructor requires the child node that this inverter  decorator
     * wraps*/
    public Inverter(BTNode node)
    {
        m_node = node;
    }

    /* Reports a success if the child fails and
     * a failure if the child succeeeds. Running will report
     * as running */
    public override NodeStates Evaluate()
    {
        switch (m_node.Evaluate())
        {
            case NodeStates.FAILURE:
                m_nodeState = NodeStates.SUCCESS;
                return m_nodeState;
            case NodeStates.SUCCESS:
                m_nodeState = NodeStates.FAILURE;
                return m_nodeState;
            case NodeStates.RUNNING:
                m_nodeState = NodeStates.RUNNING;
                return m_nodeState;
        }
        m_nodeState = NodeStates.SUCCESS;
        return m_nodeState;
    }
}
