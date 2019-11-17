import java.util.ArrayList;

public class NeuralNetwork {

    private ArrayList<Layer> network;

    public NeuralNetwork(int neurons, int layers) {
        network = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
            network.add(new Layer(neurons));
        }

    }

    public Vector guess(Vector input) {
        Vector current = input;
        Vector out;
        for (Layer layer : network) {
            out = layer.activations(current);
            current = out.copy();
        }
        return current;
    }

    public Layer get(int i) {
        return network.get(i);
    }
}
