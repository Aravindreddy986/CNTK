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

    class StrongComponentDetector
    {
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
        struct StrongComponent
        {
            TNode m_root;
            size_t m_loopId;
            std::vector<TNode> m_nestedNodes;

            bool Contains(const TNode& node)
            {
                return std::find(m_nestedNodes.begin(), m_nestedNodes.end(), node) != m_nestedNodes.end();
            }
        };

        // This function analysis the networks for recurrent loops present in the computation of 'rootNode.'
        // This sets/updates:
        //  - m_allSEQNodes
        //  - ComputationNode::m_isPartOfLoop (exposed to outside as IsPartOfLoop())
        //  - the cached m_evalOrders[root], reordered to make nodes belonging to the same loop consecutive. TODO: Try not to do that.
        // Is often called before ValidateNetwork() on a root; will be called from inside ValidateNetwork() as well.
        // This function is called for multiple nodes, e.g. eval and training criterion. I.e. it must be able to add to a previous result. E.g. it does not clear the m_visited flags at start.
        // Note: This function is not lazy, i.e. not cached. BuildAndValidateSubNetwork() caches, but others don't.
        // TODO: In the future, this function will not take a rootNode parameter anymore.

        template<class TNode>
        std::vector<StrongComponent<TNode>> StrongComponents(const std::vector<TNode>& roots, ExecutionGraph<TNode>& graph)
        {
            // Note: This is only used for resetting the state and resetting m_visitedOrder. I think we only need the set, not the order.
            auto nodes = EvaluationSort(roots, graph);

            std::map<TNode, StrongComponentNodeState> state;
            std::vector<StrongComponent<TNode>> result;

            std::stack<TNode> nodeStack;
            size_t index = 0;
            for (auto& root : roots)
            {
                if (state[root].m_visited)
                    continue;

                StrongComponentsImpl(root, nodeStack, index, state, result);
            }
        }

        template<class TNode>
        void StrongComponentsImpl(
            ExecutionGraph<TNode>& graph,
            const TNode& node,
            std::stack<TNode>& nodeStack,
            size_t& index,
            std::map<TNode, StrongComponentNodeState>& state,
            std::vector<StrongComponent<TNode>> connectedComponents)
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
                    DetermStrongComponentsImpl(graph, predecessor, nodeStack, index, state);
                    state[node].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_minIndex);
                }
                else if (state[predecessor].m_inStack)
                {
                    // successor w is in stack S and hence in the current SCC
                    // There was a bug in original BS here!
                    node[state].m_minIndex = std::min(state[node].m_minIndex, state[predecessor].m_index);
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

                if (nestedNodes.size() <= 1) // not a loop
                    return;

                // Check that node belongs only to a single connected component.
                for (const auto &scc : connectedComponents)
                {
                    for (const auto& nested : scc.m_nestedNodes)
                    {
                        if (nested != node)
                            continue;
                        LogicError("Node is participating in two different connected components, probably error of the algorithm.");
                    }
                }

                StrongComponent<TNode> strongComponent;
                strongComponent.m_root = node;
                strongComponent.m_loopId = connectedComponents.size();
                // TODO: can we prove that 'cur' == nestedNodes.front()? If so, we won't need to store it separately.
                strongComponent.m_nestedNodes = std::move(nestedNodes);
                connectedComponents.push_back(strongComponent);
            }
        }

        // Sorts nodes inside strong components according to their evaluation order.
        template<class TNode>
        inline void EvaluationSort(std::vector<StrongComponent<TNode>>& strongComponents, const ExecutionGraph<TNode>& graph, std::function<bool(const TNode&)> delayed)
        {
            // Perform reordering of loop nodes.
            std::map<TNode, bool> hasNonDelayAsParent;
            for (auto& component : strongComponents)
            {
                // Marks nodes that have non delay parent in the loop.
                const auto& nestedNodes = component.m_nestedNodes;
                for (const auto& node : nestedNodes)
                {
                    if (delayed(node))
                        continue;

                    for (auto& predecessor : graph.Predecessors(node))
                    {
                        if (std::find(nestedNodes.begin(), nestedNodes.end(), predecessor))
                            hasNonDelayAsParent[predecessor] = true;
                    }
                }

                component->m_nestedNodes = LoopEvaluationSort(component->m_nestedNodes, hasNonDelayAsParent);
            }
        }

        template<class TNode>
        void LoopEvaluationSortImpl(std::set<TNode>& visited,
            std::set<TNode>& infinite,
            std::list<TNode>& result,
            TNode node,
            std::map<TNode, bool> hasNonDelayAsParent,
            ExecutionGraph<TNode>& graph,
            StrongComponent<TNode>& component)
        {
            if (visited.find(cur) == visited.end())
            {
                visited.insert(cur);
                infinite.insert(cur); // recStack is only used for detecting infinite loops below (didn't we already discover that?)

                if (!hasNonDelayAsParent[node]) // stop when see recurrence.
                {
                    for (const auto& p : graph.Predecessors(node))
                    {
                        if (component.Contains(p))
                            LoopEvaluationSortImpl(visited, infinite, result, p, hasNonDelayAsParent, graph, component);
                    }
                }

                infinite.erase(node);
                result.push_back(node);
            }
            else if (infinite.find(node) != infinite.end())
                LogicError("Node %ls %ls operation is part of an infinite loop that cannot be unrolled.", cur->NodeName().c_str(), cur->OperationName().c_str());
        }

        // Creates the processing order within a recurrent loop.
        // Re-traverses the set of nodes between 'node' and the first delay node on each sub-graph.
        template<class TNode>
        std::list<TNode>& LoopEvaluationSort(const std::vector<TNode>& nodes,
            const std::map<TNode, bool>& hasNonDelayAsParent,
            ExecutionGraph<TNode>& graph)
        {
            // Reorder the nodes inside the loop for the execution order.
            // Each chain between two delay nodes gets reordered.
            std::list<TNode> reordered;
            std::set<TNode> visited;
            for (const auto& node : nodes)
            {
                std::set<TNode> checkInfinity;
                if (visited.find(node) == visited.end() && !hasNonDelayAsParent[node])
                    LoopEvaluationSortImpl(visited, checkInfinity, reordered, node, hasNonDelayAsParent, graph);

                if (!checkInfinity.empty())
                    LogicError("A with node %ls, loop contains no delay node.", ToString(node).c_str());
            }
        }

        // Sorts all nodes of the graph in the evaluation order given by the root nodes.
        template<class TNode>
        inline std::list<TNode> EvaluationSort(const std::vector<TNode>& roots, const ExecutionGraph<TNode>& graph, std::function<bool(const TNode&)> delayed)
        {
            StrongComponentDetector detector;
            auto strongComponents = detector.StrongComponents(roots, graph);

            auto nodes = PreOrderTraversal(roots, graph);
            if (strongComponents.empty())
                return nodes;

            EvaluationSort(strongComponents, graph, delayed);

            // Now we need to collect all strong components and the rest of the nodes
            // in the global evaluation order.
            size_t i = 0;
            std::set<TNode> seen;
            std::vector<std::pair<size_t, size_t, TNode>> globalOrder;
            for (const auto& node : nodes)
            {
                if (seen.find(node) != seen.end())
                    continue;

                auto component = std::find_if(strongComponents.begin(), strongComponents.end(),
                    [&node](StrongComponent<TNode>& c) { return c.Contains(node); });

                if (component == strongComponents.end())
                {
                    globalOrder.push_back(std::make_pair(i++, 0, node));
                    seen.insert(node);
                }
                else
                {
                    size_t j = 0;
                    for (const auto& node : component->m_nestedNodes)
                    {
                        globalOrder.push_back(std::make_pair(i, j++, node));
                        seen.insert(node);
                    }
                    i++;
                }
            }

            // Sort among nodes and inside the loops.
            std::sort(globalOrder.begin(), globalOrder.end(),
                [](const std::pair<size_t, size_t, TNode>& a, const std::pair<size_t, size_t, TNode>& b)
                { return a.item1 < b.item1 || a.item1 == b.item1 && a.item2 < b.item2; });

            // Copy end result.
            std::list<TNode> result;
            for (const auto n : globalOrder)
                result.insert(result.end(), n.item3);
            return result;
        }
    };
}