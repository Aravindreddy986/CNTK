//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <list>
#include <set>

// Currently this is refactoring of existing legacy code.
// In the future we should consider using Boost::Graph instead, but this will require more testing
// in order not to break current behavior/baselines.
namespace CNTK
{
    template<class TNode>
    class DirectedGraph
    {
    public:
        virtual std::vector<TNode> Predecessors(const TNode& node) const = 0;
    };

    namespace Internal
    {
        template<class TNode>
        static void PreOrderTraversalImpl(const TNode& root, const DirectedGraph<TNode>& graph, std::set<TNode>& visited, std::list<TNode>& result)
        {
            if (visited.find(root) != visited.end())
                return;
            visited.insert(root);

            for (const auto& node : graph.Predecessors(root))
                PreOrderTraversalImpl(node, graph, visited, result);

            result.push_back(root);
        }
    }

    template<class TNode>
    inline std::list<TNode> PreOrderTraversal(const std::vector<TNode>& roots, const DirectedGraph<TNode>& graph)
    {
        std::list<TNode> result;
        std::set<TNode> visited;
        for (const auto& root : roots)
            Internal::PreOrderTraversalImpl(root, graph, visited, result);
        return result;
    }
}