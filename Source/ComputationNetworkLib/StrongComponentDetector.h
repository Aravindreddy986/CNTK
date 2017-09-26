//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <list>
#include <set>
#include <stack>
#include <map>
#include <algorithm>
#include <assert.h>
#include <functional>
#include "GraphOperations.h"

namespace CNTK
{
    namespace Internal
    {
    }

    template<class TNode>
    struct StrongComponent
    {
        StrongComponent(const TNode& root, size_t loopId, const std::vector<TNode>&& nodes) :
            m_root(root),
            m_loopId(loopId),
            m_nodes(std::move(nodes))
        {}

        const TNode& Root() const
        {
            return m_root;
        }

        size_t LoopId() const
        {
            return m_loopId;
        }

        const std::vector<TNode>& Nodes() const
        {
            return m_nodes;
        }

        void UpdateNodeOrder(std::vector<TNode>&& nodes)
        {
            assert(std::set<TNode>(m_nodes.begin(), m_nodes.end()) == std::set<TNode>(nodes.begin(), nodes.end()));
            m_nodes = std::move(nodes);
        }

        bool Contains(const TNode& node) const
        {
            return std::find(m_nodes.begin(), m_nodes.end(), node) != m_nodes.end();
        }

    private:
        TNode m_root;
        size_t m_loopId;
        std::vector<TNode> m_nodes;
    };

    class StrongComponentDetector
    {
    public:
        // Additional information needed for Tarjan algorithm for
        // performing strong component search.
        // Same as in wikipedia, please see https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        struct StrongComponentNodeState
        {
            StrongComponentNodeState()
            {
                m_visitedOrder = -1;
                m_numNonDelayedParentsInLoop = 0;
                m_visited = false;
                m_index = -1;
                m_minIndex = -1;
                m_inStack = false;
            }

            int m_visitedOrder; // remembers order in which nodes were visited by EnumerateNodes(), but gets updated
            bool m_visited;     // note: also used by ValidateSubNetwork()
            int m_numNonDelayedParentsInLoop; // only used inside DetermineSCCs():
            int m_index;    // index denoting order in which nodes were visited in DetermineSCCs()
            int m_minIndex; // min of m_index over all nodes within a single loop
            bool m_inStack;
        };

        template<class TNode>
        void StrongComponentsImpl(
            DirectedGraph<TNode>& graph,
            const TNode& node,
            std::stack<TNode>& nodeStack,
            size_t& index,
            std::map<TNode, StrongComponentNodeState>& state,
            std::vector<StrongComponent<TNode>>& connectedComponents)
        {
            assert(!state[node].m_visited);

            // set the index (in order of visitation)
            // Each node is assigned a unique integer m_index, which numbers the nodes consecutively in the order in which they are discovered.
            state[node].m_index = index;    // TODO: can this be used as m_visitedOrder?
            state[node].m_minIndex = index; // also set m_minIndex
            index++;

            state[node].m_visited = true;

            // The nodes are placed on the stack in the order in which they are visited.
            // When the depth-first search recursively explores a node 'node' and its descendants,
            // those nodes are not all necessarily popped from the stack when this recursive call returns.
            // The crucial invariant property is that a node remains on the stack after exploration if and only if it has a path to some node earlier on the stack.
            // At the end of the call that explores 'node' and its descendants, we know whether 'node' itself has a path to any node earlier on the stack.
            // If so, the call returns, leaving 'node' on the stack to preserve the stack invariant.
            // If not, then 'node' must be the root of its strongly connected component, which consists of 'node' together with any later nodes on the stack
            // (such nodes all have paths back to 'node' but not to any earlier node,
            // because if they had paths to earlier nodes then 'node' would also have paths to earlier nodes which is false).
            // This entire component is then popped from the stack and returned, again preserving the invariant. [Wikipedia]
            nodeStack.push(node);
            state[node].m_inStack = true;

            // set m_minIndex to min over m_minIndex of children
            // m_minIndex (lowlink in Tarjan's notation) represents (roughly speaking) the smallest index of any node known to be reachable from 'node', including 'node' itself. [Wikipedia]
            for (const auto& predecessor : graph.Predecessors(node))
            {
                if (!state[predecessor].m_visited)
                {
                    // predecessor w has not yet been visited; recurse on it
                    StrongComponentsImpl(graph, predecessor, nodeStack, index, state, connectedComponents);
                    state[node].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_minIndex);
                }
                else if (state[predecessor].m_inStack)
                {
                    // successor w is in stack S and hence in the current SCC
                    // NOTE! This is actually different from the wikipedia algorithm
                    state[node].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_minIndex);
                }
            }

            // if 'node' is a root node, then we closed a loop.
            // 'node' must be left on the stack if m_minIndex < m_index,
            // whereas it must be removed as the root of a strongly connected component if m_minIndex == m_index.
            // m_minIndex is computed during the depth-first search from 'cur' (above), as this finds the nodes that are reachable from 'cur'. [Wikipedia]
            assert(state[node].m_minIndex <= state[node].m_index);
            if (state[node].m_minIndex == state[node].m_index) // m_minIndex is still equal to m_index, as we set it at the start of this function: we closed a loop
            {
                // gather the list of all nodes in this loop
                std::vector<TNode> nestedNodes;

                for (;;)
                {
                    TNode current = nodeStack.top();
                    nodeStack.pop();

                    state[current].m_inStack = false;
                    nestedNodes.push_back(current);

                    if (current == node) // hit our starting point: done
                        break;
                }

                // not a real loop. In degenerate situation it could be that the delay
                // feeds directly into itself though, but then its still just returns the same value
                // so can be evaluated in a topological sort order.
                if (nestedNodes.size() <= 1)
                    return;

                connectedComponents.emplace_back(node, connectedComponents.size(), std::move(nestedNodes));
            }
        }

        template<class TNode>
        std::vector<StrongComponent<TNode>> StrongComponents(const std::vector<TNode>& roots, DirectedGraph<TNode>& graph)
        {
            // Note: This is only used for resetting the state and resetting m_visitedOrder. I think we only need the set, not the order.
            auto nodes = PreOrderTraversal(roots, graph);

            std::map<TNode, StrongComponentNodeState> state;
            std::vector<StrongComponent<TNode>> result;

            std::stack<TNode> nodeStack;
            size_t index = 0;
            for (auto& root : roots)
            {
                if (state[root].m_visited)
                    continue;
                StrongComponentsImpl(graph, root, nodeStack, index, state, result);
            }

            return result;
        }

        // Sorts nodes inside strong components according to their evaluation order.
        // The algorithm:
        //  - take component
        //  - finds all its nodes that feed only into delay node
        //  - these nodes become new roots
        //  - perform the topological sort using these roots
        //  - update the component with the reordered list.
        template<class TNode>
        inline void EvaluationSort(std::vector<StrongComponent<TNode>>& strongComponents, const DirectedGraph<TNode>& graph, std::function<bool(const TNode&)> delay)
        {
            for (auto& component : strongComponents)
            {
                // Get all nodes that only have a delay child, these
                // will become new roots for evaluation.
                const auto& nestedNodes = component.Nodes();
                std::set<TNode> newRoots(nestedNodes.begin(), nestedNodes.end());
                for (const auto& node : nestedNodes)
                {
                    if (delay(node))
                        continue;

                    for (const auto& predecessor : graph.Predecessors(node))
                    {
                        if (component.Contains(predecessor))
                            newRoots.erase(predecessor);
                    }
                }

                // Perform the topological sort stopping at delay nodes
                // to break the loops.
                std::vector<TNode> reordered;
                reordered.reserve(component.Nodes().size());

                std::set<TNode> visited;
                for (const auto& root : newRoots)
                {
                    if (visited.find(root) != visited.end())
                        continue;

                    std::set<TNode> checkInfinity;
                    LoopEvaluationSort(visited, checkInfinity, reordered, root, graph, component, delay);
                }

                // Update the component.
                component.UpdateNodeOrder(std::move(reordered));
            }
        }

        // Creates the processing order within a recurrent loop.
        // Re-traverses the set of nodes between 'node' and the first delay node on each sub-graph.
        template<class TNode>
        void LoopEvaluationSort(std::set<TNode>& visited,
            std::set<TNode>& nodesOnThePathFromRoot,
            std::vector<TNode>& result,
            TNode node,
            const DirectedGraph<TNode>& graph,
            const StrongComponent<TNode>& component,
            std::function<bool(const TNode&)> delay)
        {
            if (visited.find(node) != visited.end())
            {
                // Check if we have a loop without a delay node.
                if (nodesOnThePathFromRoot.find(node) != nodesOnThePathFromRoot.end())
                    LogicError("Node operation is part of an infinite loop that cannot be unrolled.");
                return;
            }

            visited.insert(node);
            nodesOnThePathFromRoot.insert(node);

            // Recurse if not a delay, stop when see a recurrence.
            if (!delay(node))
            {
                for (const auto& p : graph.Predecessors(node))
                {
                    if (component.Contains(p))
                        LoopEvaluationSort(visited, nodesOnThePathFromRoot, result, p, graph, component, delay);
                }
            }

            nodesOnThePathFromRoot.erase(node);
            result.push_back(node);
        }

        // Sorts all nodes of the graph in the evaluation order given by the root nodes.
        template<class TNode>
        inline std::vector<TNode> GlobalEvaluationSort(const std::vector<StrongComponent<TNode>>& strongComponents, const std::vector<TNode>& roots, const DirectedGraph<TNode>& graph)
        {
            auto nodes = PreOrderTraversal(roots, graph);
            if (strongComponents.empty())
                return std::vector<TNode>(nodes.begin(), nodes.end());

            // Now we need to collect all strong components and the rest of the nodes
            // in the global evaluation order.

            // Prepare additional structure that contains the number of nodes per
            // component.
            std::map<std::vector<StrongComponent<TNode>>::const_iterator, size_t> componentToNodeCount;
            for (auto i = strongComponents.begin(); i != strongComponents.end(); ++i)
                componentToNodeCount.insert(std::make_pair(i, i->Nodes().size()));

            // Strong components should already be sorted in a proper evaluation order.
            // The whole strong component gets evaluated on its last node position in the global
            // topological order list('nodes').
            std::vector<TNode> result;
            result.reserve(nodes.size());
            for (const auto& node : nodes)
            {
                auto component = std::find_if(strongComponents.begin(), strongComponents.end(),
                    [&node](const StrongComponent<TNode>& c) { return c.Contains(node); });
                if (component == strongComponents.end())
                {
                    result.push_back(node);
                }
                else
                {
                    // Check if the last node of the component in the global topological
                    // sort order. If that is the case, insert all nodes of the component.
                    assert(componentToNodeCount[component] > 0);
                    if (--componentToNodeCount[component] == 0)
                        result.insert(result.end(), component->Nodes().begin(), component->Nodes().end());
                }
            }
            return result;
        }
    };
}