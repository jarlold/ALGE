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

