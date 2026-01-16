#include "node_calculus.cpp"
#include "multidimensional.cpp"
#include "optimizers.cpp"
#include "packing.cpp"
#include "common.cpp"

void memBatchTest() {
	// These will store input data
	std::vector<std::vector<NumKin>> x(100, std::vector<NumKin>(3, 0.0f));

	// This will store the output
	std::vector<NumKin> y(100);

	// Finally let's initialize them
	for (int i = 0; i < 100; i++) {
		x[i][0] = randomFloat(-1.0, 1.0);
		x[i][1] = randomFloat(-1.0, 1.0);
		x[i][2] = randomFloat(-1.0, 1.0);

		y[i] = (x[i][0] + 1) * x[i][1] + 0.5 * x[i][2];
	}

	// Then let's define a simple 3 layer neural network
	Matrix w1 = randomMatrix(5, 3);
	Matrix w2 = randomMatrix(10, 5);
	Matrix w3 = randomMatrix(1, 10);

	// Then we'll overwrite all that with a bunch of ones
	xavierInitMatrix(w1);
	xavierInitMatrix(w2);
	xavierInitMatrix(w3);

	// Model loss is a function using w1-3, which builds a graph leading to
	// the single Node computing the loss of the example.
	ModelLoss<std::vector<NumKin>, NumKin> modelLoss = [w1, w2, w3, y] (std::vector<NumKin>& inp, NumKin gt) -> NodePtr {
		// We'll have to convert from primitive to graph item
		NodePtr groundTruth = constantNode(gt);
		Vector inputs(3);
		inputs[0] = constantNode(inp[0]);
		inputs[1] = constantNode(inp[1]);
		inputs[2] = constantNode(inp[2]);

		// Build the graph for the model
		Vector outputs;
		outputs = multMatrixVector(w1, inputs);
		outputs = tanhVector(outputs);
		outputs = multMatrixVector(w2, outputs);
		outputs = tanhVector(outputs);
		outputs = multMatrixVector(w3, outputs);

		// Build the graph for the error.
		NodePtr instanceLoss = addNodes(groundTruth, negNode(outputs[0]));
		instanceLoss = multNodes(instanceLoss, instanceLoss);
		instanceLoss = divNodes(instanceLoss, constantNode(100.0f));

		// And that's how we build a graph for just one example!
		return instanceLoss;
	};

	// Function to call after forwards & backwards to update whatever weights
	// are allowed to be updated.
	ModelUpdateFunction updateWeights = [w1, w2, w3] (float lr) -> void {
		updateWeightsMatrix(w1, lr);
		updateWeightsMatrix(w2, lr);
		updateWeightsMatrix(w3, lr);
		
		clearWeightsMatrix(w1);
		clearWeightsMatrix(w2);
		clearWeightsMatrix(w3);
	};

	compressedMiniBatchGradientDescent<std::vector<NumKin>, NumKin>(
	    modelLoss,
	    x,
	    y,
	    updateWeights,
	    50,
	    10,
	    0.01f
	);
	
	
}

void memBatchTestNoPrefab() {
	// These will store input data
	std::vector<std::vector<NumKin>> x(100, std::vector<NumKin>(3, 0.0f));

	// This will store the output
	std::vector<NumKin> y(100);

	// Finally let's initialize them
	for (int i = 0; i < 100; i++) {
		x[i][0] = randomFloat(-1.0, 1.0);
		x[i][1] = randomFloat(-1.0, 1.0);
		x[i][2] = randomFloat(-1.0, 1.0);

		y[i] = (x[i][0] + 1) * x[i][1] + 0.5 * x[i][2];
	}

	// Then let's define a simple 3 layer neural network
	Matrix w1 = randomMatrix(5, 3);
	Matrix w2 = randomMatrix(10, 5);
	Matrix w3 = randomMatrix(1, 10);

	// Then we'll overwrite all that with a bunch of ones
	xavierInitMatrix(w1);
	xavierInitMatrix(w2);
	xavierInitMatrix(w3);

	// Model loss is a function using w1-3, which builds a graph leading to
	// the single Node computing the loss of the example.
	ModelLoss<std::vector<NumKin>, NumKin> modelLoss = [w1, w2, w3, y] (std::vector<NumKin>& inp, NumKin gt) -> NodePtr {
		// We'll have to convert from primitive to graph item
		NodePtr groundTruth = constantNode(gt);
		Vector inputs(3);
		inputs[0] = constantNode(inp[0]);
		inputs[1] = constantNode(inp[1]);
		inputs[2] = constantNode(inp[2]);

		// Build the graph for the model
		Vector outputs;
		outputs = multMatrixVector(w1, inputs);
		outputs = tanhVector(outputs);
		outputs = multMatrixVector(w2, outputs);
		outputs = tanhVector(outputs);
		outputs = multMatrixVector(w3, outputs);

		// Build the graph for the error.
		NodePtr instanceLoss = addNodes(groundTruth, negNode(outputs[0]));
		instanceLoss = multNodes(instanceLoss, instanceLoss);
		instanceLoss = divNodes(instanceLoss, constantNode(100.0f));

		// And that's how we build a graph for just one example!
		return instanceLoss;
	};

	// Now let's try optimizing them with the memory efficient optimizers.
	int batchSize = 10;
	for (int epoch = 0; epoch < 50; epoch++) {
		for (int batch = 0; batch < 10; batch++) {
			// Stores the graph for the current example only. We're only loading into it right
			// now so that we can get the size of the graph.
			NodePtr G = modelLoss(x[batch * batchSize], y[batch * batchSize]);
			int graphSize = sizeOfGraph(G);

			// Then we'll store all the forwards pass values into here
			ValueMatrix v(batchSize, ValueVector(graphSize));

			// And the final result of every forwards pass into here
			ValueVector subGraphLosses(batchSize);

			// First we have to compute the forwards pass values of
			// each subgraph (one at a time).
			for (int example = 0; example < batchSize; example++) {
				// We're rebuilding the graph here instead of re-loading the
				// inputs to it. That's because the template doesn't have a
				// "load graph inputs" function.
				G = modelLoss(x[batch * batchSize + example], y[batch * batchSize + example]);
				subGraphLosses[example] = forward(G);
				saveForwardsAsVector(v[example], G);
			}

			// Now that we've computed all the forward pass value, we can go ahead
			// and do the backwards pass... carefully....
			for (int example = 0; example < batchSize; example++) {
				// Sum up all the errors of everything BUT the graph of the current
				// example we're working with, that will have to be stored fully in memory.
				NodePtr errSum = constantNode(0.0f);
				for (int x = 0; x < batchSize; x++) {
					if (x == example) continue;
					errSum = addNodes(errSum, constantNode(subGraphLosses[x]));
				}

				// Then add the graph of the example we're currently working on
				NodePtr subgraph = modelLoss(x[batch * batchSize + example], y[batch * batchSize + example]);
				//forward(subgraph);
				loadForwardsFromVector(v[example], subgraph);
				errSum = addNodes(
				        errSum,
				        subgraph
				    );

				// Then we can do his backwards pass.
				backpropagation(errSum);
			}

			/* The question here is: How do we update w1..w3 with those gradients of w1..w3 we just computed.
					Answer: I think we can just call updateWeightsMatrix or updateWeights and it should be okay.
			
			   I guess the other question is "are those gradient any good anyway?"
				    Answer: No, but I don't know why.
			
			   TODO: Make them good. 
			
			*/
			float lr = 0.01;
			
			updateWeightsMatrix(w1, lr);
			updateWeightsMatrix(w2, lr);
			updateWeightsMatrix(w3, lr);
			
			clearWeightsMatrix(w1);
			clearWeightsMatrix(w2);
			clearWeightsMatrix(w3);

			// Then we'll deduce the final loss for the printout
			#ifndef IS_ARDUINO
			float curLoss = 0.0f;
			for (int example = 0; example < batchSize; example++) curLoss += subGraphLosses[example];
			if (VERBOSE_MODE) printf("MSE for epoch %d on batch %d is %f\r\n", epoch, batch, curLoss);
			#endif
		}
	}
}

void printShit(NodePtr root) {
	printf("%f\r\n", root->value);
	if (root->lhs) printShit(root->lhs);
	if (root->rhs) printShit(root->rhs);
}

void vectorSaveTest() {
	// Then let's define a simple 3 layer neural network
	Matrix w1 = randomMatrix(5, 3);
	Matrix w2 = randomMatrix(10, 5);
	Matrix w3 = randomMatrix(1, 10);

	// Then we'll overwrite all that with a bunch of ones
	xavierInitMatrix(w1);
	xavierInitMatrix(w2);
	xavierInitMatrix(w3);

	// We'll have to convert from primitive to graph item
	NodePtr groundTruth = constantNode(4.0);
	Vector inputs(3);
	inputs[0] = constantNode(3.0);
	inputs[1] = constantNode(2.0);
	inputs[2] = constantNode(1.0);

	// Build the graph for the model
	Vector outputs;
	outputs = multMatrixVector(w1, inputs);
	outputs = tanhVector(outputs);
	outputs = multMatrixVector(w2, outputs);
	outputs = tanhVector(outputs);
	outputs = multMatrixVector(w3, outputs);

	// Build the graph for the error.
	NodePtr instanceLoss = addNodes(groundTruth, negNode(outputs[0]));
	instanceLoss = multNodes(instanceLoss, instanceLoss);
	instanceLoss = divNodes(instanceLoss, constantNode(100.0f));
	
	// TODO: Make sure the following operation has no effect.
	ValueVector f_val(sizeOfGraph(instanceLoss));
	
	NodePtr B = instanceLoss->lhs->rhs->rhs;
	float b = B->value;
	printf("B: %f\r\n", b);
	
	forward(instanceLoss);
	printShit(instanceLoss);
	saveForwardsAsVector(f_val, instanceLoss);
	
	b = B->value;
	printf("B og: %f\r\n", b);
	
	resetGraph(instanceLoss);
	b = B->value;
	printf("B reset: %f\r\n", b);
	
	loadForwardsFromVector(f_val, instanceLoss);
	b = B->value;
	printf("B reload: %f\r\n", b);
	printf("--- BREAK ---\r\n");
	printShit(instanceLoss);
}

int main() {
}


