/*  Create a compute graph of basic operations that
    we can support the backwards propogation of. So essentially
    you can differentiate and optimize any function that you can
    describe using the operations listed in 
    the Operation enum.
*/

#pragma once
#include <stdio.h>
#include <cmath>
#include <memory>
#include <set>
#include <cassert>
#include <list>

typedef float NumKin;

enum Operation {
    MULT,
    ADD,
    POW,
    EXP,
    LOG,
    SIN,
    COS,
    TANH,
    DIV,
    RELU,
    NONE
};

struct Node {
    NumKin value = 0.0f;
    NumKin grad = 0.0f;
    Operation oper = NONE;
    std::shared_ptr<Node> lhs = nullptr;
    std::shared_ptr<Node> rhs = nullptr; // if the operation only takes
                                         // one input then this bad boy gets to stay null
    unsigned int visitId; // Stores when we visited the node, useful for debug.
	bool visited = false;
};

// A man can only type so much
using NodePtr = std::shared_ptr<Node>;

// Should really be called "variable node" in hindsight.
NodePtr constantNode(NumKin v) {
    auto n = std::make_shared<Node>();
    n->value = v;
    return n;
}

/* GRAPH OPERATIONS */
NodePtr addNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = ADD;
    return n;
}

NodePtr multNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = MULT;
    return n;
}

NodePtr powNode(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = POW;
    return n;
}

NodePtr expNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = EXP;
    return n;
}

NodePtr logNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = LOG;
    return n;
}

NodePtr sinNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = SIN;
    return n;
}

NodePtr cosNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = COS;
    return n;
}

NodePtr tanhNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = TANH;
    return n;
}

NodePtr divNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = DIV;
    return n;
}

NodePtr reluNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = RELU;
    return n;
}

NodePtr sqrtNode(NodePtr lhs) {
    auto half = constantNode(0.5f);
    return powNode(lhs, half);
}

NodePtr negNode(NodePtr lhs) {
    auto minusOne = constantNode(-1.0f);
    return multNodes(minusOne, lhs);
}

// Forward pass with visited set
// This goes through and updates all the values in the nodes
void updateValue(NodePtr n, std::set<NodePtr>& visited) {
    // Make sure we don't re-use a node somehow.
    // This shouldn't get triggered but I'm pretty sure it does...
    if (!n || visited.count(n)) return;
    visited.insert(n);

    if (n->oper == NONE) return;

    if (n->lhs) updateValue(n->lhs, visited);
    if (n->rhs) updateValue(n->rhs, visited);

    switch (n->oper) {
        case MULT:
            n->value = n->lhs->value * n->rhs->value;
            break;
        case ADD:
            n->value = n->lhs->value + n->rhs->value;
            break;
        case POW:
            n->value = pow(n->lhs->value, n->rhs->value);
            break;
        case EXP:
            n->value = exp(n->lhs->value);
            break;
        case LOG:
            n->value = log(n->lhs->value);
            break;
        case SIN:
            n->value = sin(n->lhs->value);
            break;
        case COS:
            n->value = cos(n->lhs->value);
            break;
        case TANH:
            n->value = tanh(n->lhs->value);
            break;
        case DIV:
            n->value = n->lhs->value / n->rhs->value;
            break;
        case RELU:
            n->value = std::max(0.0f, n->lhs->value);
            break;
        default:
            break;
    }
}

NumKin forward(NodePtr root) {
    //TODO: Instead of donig this recurisvely iterate without
    // revisiting. I forget the name of what its called when you
    // do that.
    std::set<NodePtr> visited;
    updateValue(root, visited);
    return root->value;
}

// Backward pass with visited set
int lll = 0;
void updateBack(NodePtr n, std::set<NodePtr>& visited) {
    if (!n || visited.count(n) > 0) return;
    n->visitId = lll;
    lll++;
    visited.insert(n);

    if (n->oper == NONE) return;

    switch (n->oper) {
        case MULT:
            n->lhs->grad += n->rhs->value * n->grad;
		
            n->rhs->grad += n->lhs->value * n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case ADD:
            n->lhs->grad += n->grad;
            n->rhs->grad += n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case POW:
            if (n->lhs->value > 0) {
                n->lhs->grad += n->rhs->value * pow(n->lhs->value, n->rhs->value - 1) * n->grad;
                n->rhs->grad += log(n->lhs->value) * pow(n->lhs->value, n->rhs->value) * n->grad;
            }
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case EXP:
            n->lhs->grad += n->value * n->grad;
            updateBack(n->lhs, visited);
            break;
        case LOG:
            n->lhs->grad += (1 / n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case SIN:
            n->lhs->grad += cos(n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case COS:
            n->lhs->grad += -sin(n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case TANH: {
            NumKin t = tanh(n->lhs->value);
            n->lhs->grad += (1 - t * t) * n->grad;
            updateBack(n->lhs, visited);
            break;
        }
        case DIV: {
            NumKin a = n->lhs->value;
            NumKin b = n->rhs->value;
            n->lhs->grad += (1 / b) * n->grad;
            n->rhs->grad += (-a / (b * b)) * n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        }
        case RELU: {
            //TODO: Does this work?
            n->lhs->grad += (n->lhs->value > 0 ? 1.0f : 0.0f) * n->grad;
            updateBack(n->lhs, visited);
            break;
        }
        default:
            break;
    }
}

void resetGraph(NodePtr root) {
    // Make the graph ready to use again!
    root->grad = 0;
	root->visited = false;
    // Reset all the values that aren't 'constants'
    if (root->oper != NONE) root->value = 0;

    // Not just the root node, but the root women and root children
    // too!
    if (root->lhs) resetGraph(root->lhs);
    if (root->rhs) resetGraph(root->rhs);
}

void backpropagation(NodePtr root) {
    root->grad = 1.0f;
    std::set<NodePtr> visited;
    updateBack(root, visited);
}

void setOfNodes(NodePtr root, std::set<NodePtr>& visited) {
	if (!root || visited.count(root)) return;
	visited.insert(root);
	
	if (root->rhs) setOfNodes(root->lhs, visited);
	if (root->lhs) setOfNodes(root->rhs, visited);
}

void listOfNodes(NodePtr root, std::list<NodePtr>& visited) {
	if (root->visited) return;
	visited.push_back(root);
	root->visited = true;
	if (root->lhs) listOfNodes(root->lhs, visited);
	if (root->rhs) listOfNodes(root->rhs, visited);
}

int sizeOfGraph(NodePtr root) {
	std::set<NodePtr> visited;
	setOfNodes(root, visited);
	return visited.size();
}
