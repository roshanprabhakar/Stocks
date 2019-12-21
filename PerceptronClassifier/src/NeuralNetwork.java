import java.util.ArrayList;

public class NeuralNetwork {

    private ArrayList<Layer> network;

    public NeuralNetwork(int neurons, int layers) {
        network = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
            network.add(new Layer(neurons));
        }
    }

    public Vector forwardProp(Vector input) {
        Vector passed = network.get(0).activations(input);
        for (int i = 1; i < network.size(); i++) {
            passed = network.get(i).activations(passed);
        }
        return passed;
    }

    public void train(Vector input, Vector correct) {
        double loss = input.loss(correct);
    }

    public void updateWeight(int layer, int perceptron, int connection, double shift) {
        network.get(layer).get(perceptron).shiftWeight(connection, shift);
    }

    public Layer get(int i) {
        return network.get(i);
    }
}
