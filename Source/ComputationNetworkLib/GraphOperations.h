//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <list>
#include <set>

namespace CNTK
{
    template<class TNode>
    class ExecutionGraph
    {
    public:
        virtual std::vector<TNode> Predecessors(const TNode& node) = 0;
    };

    namespace Internal
    {
        template<class TNode>
        static void PreOrderTraversalImpl(const TNode& root, const ExecutionGraph<TNode>& graph, std::set<TNode>& visited, std::list<TNode>& result)
        {
            if (visited.find(root) != visited.end())
                return;
            visited.insert(root);

            for (auto node : graph.Predecessors(root))
                PreOrderTraversalImpl(node, graph, visited, result);

            result.push_back(root);
        }
    }

    template<class TNode>
    inline std::list<TNode> PreOrderTraversal(const std::vector<TNode>& roots, const ExecutionGraph<TNode>& graph)
    {
        std::list<TNode> result;
        std::set<TNode> visited;
        for (const auto& root : roots)
            Internal::PreOrderTraversalImpl(root, graph, visited, result);
        return result;
    }
}