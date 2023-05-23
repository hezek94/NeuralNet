#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// NeuralNetwork.hpp
class NeuralNetwork {
private:
    std::vector<uint> topology;
    Scalar learningRate;
public:
    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;

    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector& input);

    void propagateBackward(RowVector& output);

    void calcErrors(RowVector& output);

    void updateWeights();

    void train(std::vector<RowVector*>& input_data, std::vector<RowVector*>& output_data);
};
