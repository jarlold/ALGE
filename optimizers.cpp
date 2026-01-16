#pragma once

#include <vector>
#include <functional>
#include <cstdlib>
#include "node_calculus.cpp"
#include "multidimensional.cpp"
#include "flags.cpp"
#include "packing.cpp"

// Compute the loss of a model for a given example instance (ModelInputType being the example instance)
template<typename ModelInputType, typename ModelOutputType> using ModelLoss = std::function< NodePtr(ModelInputType&, ModelOutputType&)>;

// After we do the backwards pass, this should update the model's weights.
using ModelUpdateFunction = std::function<void(float)>;

// Perform full gradient descent on all the entries in the dataset, memory be damned.
template<typename ModelInputType, typename ModelOutputType>
void batchGradientDescent(
    ModelLoss<ModelInputType, ModelOutputType>& modelLoss,
    std::vector<ModelInputType>& modelInputs, // Xs
    std::vector<ModelOutputType>& modelOutputs,  //Ys
    ModelUpdateFunction updateGrad,
    int maxIter,
    float lr
) {
	// Build the graph to find the sum of all the loss.
	NodePtr batchLoss = constantNode(0.0f);
	int numExamples = modelInputs.size();
	for (int i = 0; i < numExamples; i++) {
		batchLoss = addNodes(modelLoss(modelInputs[i], modelOutputs[i]), batchLoss);
	}

	// Then we'll try and lower the sum of the losses as much as possible.
	for (int i = 0; i < maxIter; i++) {
		backpropagation(batchLoss);
		updateGrad(lr);
		resetGraph(batchLoss);
	}
}

// Like the above, but do little mini-batches so we don't run out of memory.
template<typename ModelInputType, typename ModelOutputType>
void miniBatchGradientDescent(
    ModelLoss<ModelInputType, ModelOutputType>& modelLoss,
    std::vector<ModelInputType>& modelInputs, // Xs
    std::vector<ModelOutputType>& modelOutputs,  //Ys
    ModelUpdateFunction updateGrad,
    int maxIter,
    int batchSize,
    float lr
) {
	// If we can't fit all the data into neat batches we'll give up and make it the user's problem
	if (modelOutputs.size() % batchSize != 0 || modelInputs.size() % batchSize != 0) {
		throw 1;
	}

	// Divide all the examples into mini batches of whatever size. THIS ASSUMES THEY ARE SHUFFLED ALREADY!!!
	int numBatches = modelOutputs.size() / batchSize;
	std::vector<std::vector<ModelInputType>> batchInputs(batchSize, std::vector<ModelInputType>(batchSize));
	std::vector<std::vector<ModelOutputType>> batchOutputs(batchSize, std::vector<ModelOutputType>(batchSize));

	for (int i = 0; i < numBatches; i++) {
		for (int j = 0; j < batchSize; j++) {
			batchInputs[i][j] = modelInputs[i * batchSize + j];
			batchOutputs[i][j] = modelOutputs[i * batchSize + j];
		}
	}

	// Now we can start optimizing
	for (int epoch = 0; epoch < maxIter; epoch++) {
		for (int batch = 0; batch < numBatches; batch++) {
			// So that we don't use up so much memory, we'll have to re-build
			// the loss graph every time.
			NodePtr loss = constantNode(0.0f);
			for (int example = 0; example < batchSize; example++) {
				NodePtr e = modelLoss(batchInputs[batch][example], batchOutputs[batch][example]);
				loss = addNodes(loss, e);
			}

			// Then we can actually do like, the optimization part, lol
			NumKin curLoss = forward(loss);
			backpropagation(loss);
			updateGrad(lr);

			#ifndef IS_ARDUINO
				if (VERBOSE_MODE) printf("MSE for epoch %d on batch %d is %f\r\n", epoch, batch, curLoss);
			#endif

			// I guess we don't need to reset the graph, considering we're about to throw it out...
			//resetGraph(loss);
		}
	}
}

// Like the function above, but use compression techniques and graph re-building to
// minimize our memory footprint as much as we can- even at the cost of speed.
template<typename ModelInputType, typename ModelOutputType>
void compressedMiniBatchGradientDescent(
    ModelLoss<ModelInputType, ModelOutputType>& modelLoss,
    std::vector<ModelInputType>& modelInputs, // Xs
    std::vector<ModelOutputType>& modelOutputs,  //Ys
    ModelUpdateFunction updateGrad,
    int maxIter,
    int batchSize,
    float lr
) {
	
	
	// If we can't fit all the data into neat batches we'll give up and make it the user's problem
	if (modelOutputs.size() % batchSize != 0 || modelInputs.size() % batchSize != 0) {
		throw 1;
	}
	
	// Divide all the examples into mini batches of whatever size. THIS ASSUMES THEY ARE SHUFFLED ALREADY!!!
	int numBatches = modelOutputs.size() / batchSize;
	std::vector<std::vector<ModelInputType>> batchInputs(batchSize, std::vector<ModelInputType>(batchSize));
	std::vector<std::vector<ModelOutputType>> batchOutputs(batchSize, std::vector<ModelOutputType>(batchSize));

	for (int i = 0; i < numBatches; i++) {
		for (int j = 0; j < batchSize; j++) {
			batchInputs[i][j] = modelInputs[i * batchSize + j];
			batchOutputs[i][j] = modelOutputs[i * batchSize + j];
		}
	}

	// Now we can start optimizing
	for (int epoch = 0; epoch < maxIter; epoch++) {
		for (int batch = 0; batch < numBatches; batch++) {
			// Stores the graph for the current example only. We're only loading into it right
			// now so that we can get the size of the graph.
			NodePtr G = modelLoss(batchInputs[batch][0], batchOutputs[batch][0]);
			
			// Then we'll store all the forwards pass values into here
			std::vector<PackedVector> v(batchSize);
			ValueVector temp;
			
			// And the final result of every forwards pass into here
			ValueVector subGraphLosses(batchSize);
			
			// First we have to compute the forwards pass values of
			// each subgraph (one at a time).
			for (int example=0; example < batchSize; example++) {
				// We're rebuilding the graph here instead of re-loading the
				// inputs to it. That's because the template doesn't have a 
				// "load graph inputs" function.
				G = modelLoss(batchInputs[batch][example], batchOutputs[batch][example]);
				subGraphLosses[example] = forward(G);
				saveForwardsAsVector(temp, G);
				v[example] = packVector(temp);
			}
			
			// Now that we've computed all the forward pass value, we can go ahead
			// and do the backwards pass... carefully....
			for (int example=0; example < batchSize; example++) {
				// Sum up all the errors of everything BUT the graph of the current
				// example we're working with, that will have to be stored fully in memory.
				NodePtr errSum = constantNode(0.0f);
				for (int x=0; x < batchSize; x++) {
					if (x == example) continue;
					errSum = addNodes(errSum, constantNode(subGraphLosses[x]));
				}
				
				// Then add the graph of the example we're currently working on
				NodePtr subgraph = modelLoss(batchInputs[batch][example], batchOutputs[batch][example]);
				temp = unpackVector(v[example]);
				loadForwardsFromVector(temp, subgraph);
				errSum = addNodes(
						errSum,
						subgraph
				);
				
				// Then we can do his backwards pass.
				backpropagation(errSum);
			}
	
			updateGrad(lr);
			
			// Then we'll deduce the final loss for the printout
			#ifndef IS_ARDUINO
				float curLoss = 0.0f;
			    for (int example=0; example<batchSize; example++) curLoss += subGraphLosses[example];
				if (VERBOSE_MODE) printf("MSE for epoch %d on batch %d is %f\r\n", epoch, batch, curLoss);
			#endif
		}
	}
}




