/* This file is for extending the basic operations in
    node_calculus.cpp to support multidimensional linear
    algebra type stuff.
 */
#pragma once
#include <vector>
#include <functional>
#include "node_calculus.cpp"
#include "common.cpp"

// Matrix is a 2D vector of NodePtrs
// tensor3 is a bunch of matrices, and well,
// you get where this is going
using Vector = std::vector<NodePtr>;
using Matrix = std::vector<Vector>;
using Tensor3 = std::vector<Matrix>;
using Tensor4 = std::vector<Tensor3>;

// When they contain nothing but actual numbers, we'll
// call them "value TYPE" instead, but they're just
// multidimensional std::vectors. 
using ValueVector = std::vector<NumKin>;
using ValueMatrix = std::vector<ValueVector>;
using ValueTensor3 = std::vector<ValueMatrix>;
using ValueTensor4 = std::vector<ValueTensor3>;

/* Vectors */
// This section still needs some testing
Vector addVector(Vector& v1, Vector& v2) {
    int len = v1.size();
    Vector result(len);
    for (int i = 0; i < len; ++i) {
        result[i] = addNodes(v1[i], v2[i]);
    }
    return result;
}

Vector addVector(Vector& v1, NodePtr n) {
    // Is it okay that these are all using the same N? Probably, maybe. Idk?
    int len = v1.size();
    Vector result(len);
    for (int i = 0; i < len; ++i) {
        result[i] = addNodes(v1[i], n);
    }
    return result;
}

NodePtr sumVector(Vector& v) {
    NodePtr result;
    for (size_t i =0; i < v.size(); i++) {
        NodePtr a = addNodes(result, v[i]); 
        result = a;
    }
    return result;
}

Vector scaleVector(Vector& v1, NodePtr s) {
    int len = v1.size();
    Vector result(len);
    for (int i = 0; i < len; ++i) {
        result[i] = multNodes(v1[i], s);
    }
    return result;
}

Vector randomVector(int length) {
    Vector v(length);
    for (int i = 0; i < length; ++i) {
        v[i] = constantNode(randomFloat());
    }
    return v;
}

Vector zeroVector(int length) {
    Vector v(length);
    for (int i = 0; i < length; ++i) {
        v[i] = constantNode(0);
    }
    return v;
}

Vector tanhVector(const Vector& vec) {
    int l = vec.size(); // I'm 90% sure the compiler knows how to do this on it's own.
    Vector output = randomVector(l);
    for (int i =0; i < l; i++) {
        output[i] = tanhNode(vec[i]);
    }
    return output;
}

Vector sigmoidVector(const Vector& vec) { // TODO: TEST ME!!!
    int l = vec.size();
    Vector output = zeroVector(l);
    for (int i=0; i<l; i++) {
        // output[i] = e^(vec[i]) / ( e^(vec[i]) + 1)
        output[i] = divNodes(expNode(vec[i]), addNodes(expNode(vec[i]), constantNode(1)));    
    }
    return output;
}

/* Matrices */
Matrix randomMatrix(int length, int width) {
    Matrix m(length);
    for (int i = 0; i < length; ++i) {
        m[i] = randomVector(width);
    }
    return m;
}

Matrix zeroMatrix(int length, int width) {
    Matrix m(length);
    for (int i = 0; i < length; ++i) {
        m[i] = zeroVector(width);
    }
    return m;
}

ValueMatrix getMatrixFromGrad(const Matrix& matrix) {
    int length = matrix.size();
    int width = matrix[0].size();
	ValueMatrix m(length, ValueVector(width, 0.0f)); // do we really need to initialize?
	
    for (int i =0; i < length; i++) {
        for (int j =0; j< width; j++) {
            m[i][j] = matrix[i][j]->grad;
        }
    }

    return m;
}

Vector multMatrixVector(const Matrix& matrix, const Vector& vector) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    int vectorLength = vector.size();

    if (numCols != vectorLength) {
        throw 1;
    }

    Vector result(numRows);
    for (int i=0; i < numRows; i++) {
        // Initialize it as all zeros
        result[i] = constantNode(0);

        // Tally up the row-col multiplication
        for (int j=0; j< numCols; j++) {
            NodePtr s = multNodes(matrix[i][j], vector[j]);
            result[i] = addNodes(result[i], s);
        }
    }

    return result;
}

Matrix addMatrix(const Matrix& A, const Matrix& B) {
    int w = A.size();
    int l = A[0].size();
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw 1;
    }

    Matrix sum(w, Vector(l, 0));
    for (int i=0; i < w; i++) {
        for (int j=0; j<l; j++) {
            sum[i][j] = addNodes(A[i][j], B[i][j]);
        }
    }

    return sum;
}

Matrix multMatrix(const Matrix& A, const Matrix& B) {
    size_t m = A.size();                 // Rows in A
    size_t k = A[0].size();              // Cols in A == Rows in B
    size_t n = B[0].size();              // Cols in B

    Matrix C(m, std::vector<NodePtr>(n));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            NodePtr sum = multNodes(A[i][0], B[0][j]);
            for (size_t l = 1; l < k; ++l) {
                sum = addNodes(sum, multNodes(A[i][l], B[l][j]));
            }
            C[i][j] = sum;
        }
    }

    return C;
}

Matrix transpose(const Matrix& input) {
    if (input.empty()) {
        throw 1;
    }

    int rows = input.size();
    int cols = input[0].size();
    Matrix result(cols, Vector(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = input[i][j];
        }
    }

    return result;
}

void updateWeightsMatrix(Matrix m, float lr) {
    int w = m.size();
    int l = m[0].size();

//	printf("[\r\n");
    for (int i=0; i < w; i++) {
        for (int j=0; j<l; j++) {
            m[i][j]->value -= m[i][j]->grad * lr;
		//	printf("    %f ", m[i][j]->grad);
        }
	//	printf("\r\n");
    }
//	printf("]\r\n");
}

void clearWeightsMatrix(Matrix m) {
	int w = m.size();
	int l = m[0].size();
	
	for (int i=0; i < w; i++) {
		for (int j=0; j<l; j++) {
			m[i][j]->grad = 0;
		}
	}
}

void updateWeightsMatrixMomentum(ValueMatrix& momentum, Matrix& weights, float lr, float alpha) {
	ValueMatrix g = getMatrixFromGrad(weights);
	
	int length = weights.size();
	int width = weights[0].size();
	
	for (int i=0; i < length; i++) {
		for (int j=0; j < width; j++) {
			momentum[i][j] = momentum[i][j]*(1 - alpha);
			momentum[i][j] += alpha*g[i][j];
			weights[i][j]->value += lr*momentum[i][j];
		}
	}    
}


void onesInitMatrix(Matrix& m) {
    int out = m.size();
    int in = m[0].size();
    for (int i=0; i < out; i++) {
        for (int j=0; j<in; j++) {
            m[i][j]->value = 1.0;
        }
    }
}

void xavierInitMatrix(Matrix& m) {
    int out = m.size();
    int in = m[0].size();

    NumKin limit = std::sqrt(6.0f / (in + out));

    for (int i=0; i < out; i++) {
        for (int j=0; j<in; j++) {
            m[i][j]->value = randomFloat(-limit, limit);
        }
    }
}

// This is slow and meant only for debugging
bool matrixGradIsNotZero(const Matrix& m) {
    int l = m.size();
    int w = 1;
    if (l > 0) w = m[0].size();

    for (int i=0; i<l; i++) {
        for (int j=0; j<w; j++) {
            if (m[i][j]->grad != 0.0) {
                return true;
            }
        }
    }
    return false;
}

/* Tensors*/
Tensor3 randomTensor3(int length, int width, int height) {
    Tensor3 t(length);
    for (int i = 0; i < length; ++i) {
        t[i] = randomMatrix(width, height);
    }
    return t;
}

Tensor4 randomTensor4(int length, int width, int height, int depth) {
    Tensor4 t(length);
    for (int i = 0; i < length; ++i) {
        t[i] = randomTensor3(width, height, depth);
    }
    return t;
}

Tensor3 padTensor3(const Tensor3& input, int pad_h, int pad_w, const NodePtr& zero_node) {
    // input: C x H x W
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int H_padded = H + 2 * pad_h;
    int W_padded = W + 2 * pad_w;

    Tensor3 output(C, Matrix(H_padded, Vector(W_padded, zero_node)));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h + pad_h][w + pad_w] = input[c][h][w];
            }
        }
    }
    return output;
}

Tensor4 padTensor4(const Tensor4& input, int pad) {
    if (pad <= 0) return input;

    size_t padding = static_cast<size_t>(pad);

    size_t batch_size = input.size();
    size_t channels = input[0].size();
    size_t height = input[0][0].size();
    size_t width = input[0][0][0].size();

    size_t padded_height = height + 2 * padding;
    size_t padded_width = width + 2 * padding;

    Tensor4 padded(batch_size, Tensor3(channels, Matrix(padded_height, Vector(padded_width))));

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < padded_height; ++y) {
                for (size_t x = 0; x < padded_width; ++x) {
                    if (y < padding ||
                        y >= padding + height ||
                        x < padding ||
                        x >= padding + width
                    ) {
                        // Pad with zeros
                        padded[b][c][y][x] = constantNode(0.0f);
                    } else {
                        padded[b][c][y][x] = input[b][c][y - padding][x - padding];
                    }
                }
            }
        }
    }
    return padded;
}


/* Functions relating to convultional neural networks*/
Matrix im2col(const Tensor3& input, int kernel_h, int kernel_w, int stride = 1) {
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int out_h = (H - kernel_h) / stride + 1;
    int out_w = (W - kernel_w) / stride + 1;

    int patch_size = C * kernel_h * kernel_w;
    int num_patches = out_h * out_w;

    Matrix cols(patch_size, Vector(num_patches));

    int patch_idx = 0;
    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            int col_pos = 0;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_idx = i * stride + kh;
                        int w_idx = j * stride + kw;
                        cols[col_pos][patch_idx] = input[c][h_idx][w_idx];
                        ++col_pos;
                    }
                }
            }
            ++patch_idx;
        }
    }
    return cols;
}

Tensor4 conv2d(const Tensor4& input, const Tensor4& filters, int stride = 1, int padding = 0) {
    int batch_size = input.size();
    int out_channels = filters.size();
    int in_channels = input[0].size();
    int kernel_h = filters[0][0].size();
    int kernel_w = filters[0][0][0].size();

    // Pad input
    Tensor4 padded_input = padTensor4(input, padding);

    // Determine output spatial dimensions
    int out_h = (padded_input[0][0].size() - kernel_h) / stride + 1;
    int out_w = (padded_input[0][0][0].size() - kernel_w) / stride + 1;

    Tensor4 output(batch_size, Tensor3(out_channels, Matrix(out_h, Vector(out_w))));

    for (int b = 0; b < batch_size; ++b) {
        // im2col for the current input
        // shape: (out_h * out_w, in_channels * kernel_h * kernel_w)
        Matrix cols = im2col(padded_input[b], kernel_h, kernel_w, stride);  

        for (int oc = 0; oc < out_channels; ++oc) {
            // Flatten filter to a row vector
            Matrix filter_row(1);
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        filter_row[0].push_back(filters[oc][ic][kh][kw]);
                    }
                }
            }

            // Matrix multiplication: (1 x K) * (K x N) = (1 x N)
            Matrix result = multMatrix(filter_row, transpose(cols));  // shape: (1, out_h * out_w)

            // Reshape result into output matrix
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    output[b][oc][i][j] = result[0][i * out_w + j];
                }
            }
        }
    }

    return output;
}

Vector softmax(Vector& logits) {
    Vector probs(logits.size());

    // S = sum e^x_i
    NodePtr sum = constantNode(0);
    for (size_t i=0; i<logits.size(); i++) {
        sum = addNodes(sum, expNode(logits[i]));
    }

    // Probs_i = x_i / S 
    for (size_t i=0; i<logits.size(); i++) {
        probs[i] = expNode(logits[i]);
    }

    for (size_t i=0; i < logits.size(); i++) {
        probs[i] = divNodes(probs[i], sum);
    }

    return probs;
}

