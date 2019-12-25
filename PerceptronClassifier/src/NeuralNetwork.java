import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork implements NetworkConstants {

    private ArrayList<Layer> network;
    private int neuronsPerLayer;
    private int layers;

    public NeuralNetwork(int neurons, int layers) {
        network = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
            if (i == 0) network.add(new Layer(neurons, DATA_LENGTH));
            else network.add(new Layer(neurons, neurons));
        }

        this.neuronsPerLayer = neurons;
        this.layers = layers;
    }

    public ForwardPropOutput forwardProp(Vector input) {
        Vector[] matrix = new Vector[network.size()];
        Vector passed = network.get(0).activations(input);
        matrix[0] = passed.copy();
        for (int i = 1; i < network.size(); i++) {
            passed = network.get(i).activations(passed);
            matrix[i] = passed.copy();
        }
        return new ForwardPropOutput(passed, matrix);
    }

    public void train(Vector input, Vector correct) {
        ForwardPropOutput output = forwardProp(input);
        Vector prediction = output.getResultant();
        double loss = prediction.loss(correct);

        Vector[] neuronActivations = output.getIntermediaryMatrix();
        Vector[][] imWeights = getWeightDerivatives(neuronActivations, input);
        Vector[][] imLayers = getLayerDerivatives(neuronActivations);
    }

    public Vector[][] getWeightDerivatives(Vector[] neuronActivations, Vector input) {
        Vector[][] weightDerivatives = new Vector[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            System.out.println("----------------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println();
            System.out.println("----------------------------------");
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                System.out.println("neuron: " + neuron);

                Vector pDerivatives;
                double guess;
                if (layer == 0) {

                    pDerivatives = new Vector(input.length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(input);

                    for (int weight = 0; weight < input.length(); weight++) {
                        System.out.println("weight: " + weight);
                        double factor = input.get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }

                } else {

                    pDerivatives = new Vector(neuronsPerLayer);
                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        System.out.println("weight: " + weight);
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }
                }

                System.out.println("pDerivatives: " + pDerivatives);
                System.out.println("appending pDerivatives to layer: " + layer + ", neuron: " + neuron);
                weightDerivatives[layer][neuron] = pDerivatives;
                System.out.println();
            }
        }

        //display appropriate derivative information
        for (int layer = 0; layer < weightDerivatives.length; layer++) {
            for (int neuron = 0; neuron < weightDerivatives[layer].length; neuron++) {
                System.out.print(weightDerivatives[layer][neuron] + ",  ");
            }
            System.out.println();
        }
        System.out.println();
        return weightDerivatives;
    }

    //first null layer in return matrix represents the unusability of the loss derivative with respect to the input vector
    public Vector[][] getLayerDerivatives(Vector[] neuronActivations) {
        Vector[][] layerDerivatives = new Vector[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(neuronsPerLayer);
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                }
                layerDerivatives[layer][neuron] = pDerivatives;
            }
        }
        for (int layer = 0; layer < layerDerivatives.length; layer++) {
            for (int neuron = 0; neuron < layerDerivatives[layer].length; neuron++) {
                System.out.print(layerDerivatives[layer][neuron] + ",  ");
            }
            System.out.println();
        }
        return layerDerivatives;
    }

    public void updateWeight(int layer, int perceptron, int connection, double shift) {
        network.get(layer).get(perceptron).shiftWeight(connection, shift);
    }

    public Perceptron getPerceptron(int layer, int perceptron) {
        return network.get(layer).get(perceptron);
    }

    public Layer getLayer(int i) {
        return network.get(i);
    }
}
