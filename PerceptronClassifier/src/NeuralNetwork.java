import java.beans.PropertyChangeListenerProxy;
import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork implements NetworkConstants {

    private ArrayList<Layer> network;
    private int neuronsPerLayer;
    private int inputSize; //unused but probably necessary
    private int layers;

    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers, int numOutputNeurons) {
        network = new ArrayList<>();
        for (int i = 0; i < numHiddenLayers; i++) {
            if (i == 0) network.add(new Layer(neuronsPerHidden, inputSize));
            else network.add(new Layer(neuronsPerHidden, neuronsPerHidden));
        }

//        network.add(new Layer(numOutputNeurons, neuronsPerHidden));

        this.inputSize = inputSize;
        this.neuronsPerLayer = neuronsPerHidden;
        this.layers = numHiddenLayers;
    }

    //Customized for sigmoid activations active in every neuron
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
        Vector[] neuronActivations = output.getIntermediaryMatrix();

        Vector[][] weightsDWRTLayers = getWeightDerivatives(neuronActivations, input);
//        Double[][] imBiases = getBiasDerivatives(neuronActivations, input); //TODO debug this
        Vector[][] layerDWRTpreviousLayers = getLayerDerivatives(neuronActivations);
        Vector lastLayerDWRTLoss = getDerivativeOfLossWithLastLayer(correct, neuronActivations[neuronActivations.length - 1]);

        Vector[] activationDWRTLoss = getLossDWRTactivations(lastLayerDWRTLoss, layerDWRTpreviousLayers);
        Vector[][] weightDWRTLoss = getWeightDWRTLoss(activationDWRTLoss, weightsDWRTLayers); //TODO not recognizing the multiple weights of hidden layer 0

        System.out.println("COMPLETE DERIVATIVES OF LOSS WITH RESPECT TO EACH INDIVIDUAL WEIGHT");
        for (int layer = 0; layer < layers; layer++) {
            System.out.println("----------------- LAYER: " + layer + " -----------------");
            for (int neuron = 0; neuron < neuronsPerLayer; neuron++) {
                System.out.println("----------------- NEURON: " + neuron + " -----------------");
                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
                    System.out.println(network.get(layer).get(neuron).getWeights().get(weight));
                }
            }
        }
    }

    public Vector[][] getWeightDerivatives(Vector[] neuronActivations, Vector input) {
        Vector[][] weightDerivatives = new Vector[layers][neuronsPerLayer]; //turn this to arraylist of arrays of vectors
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
//                        System.out.println("weight: " + weight + ": " + network.get(layer).get(neuron).getWeights().get(weight));
                        double factor = input.get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }

                } else {

                    pDerivatives = new Vector(neuronsPerLayer);
                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        System.out.println("weight: " + weight + ": " + network.get(layer).get(neuron).getWeights().get(weight));
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }
                }

                System.out.println("pDerivatives: " + pDerivatives);
                System.out.println("appending pDerivatives to layer: " + layer + ", neuron: " + neuron);
                weightDerivatives[layer][neuron] = pDerivatives;
                System.out.println("bias: " + network.get(layer).get(neuron).getBias());
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
        System.out.println();
        Vector[][] layerDerivatives = new Vector[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            System.out.println("----------------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("----------------------------------");
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(neuronsPerLayer);
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);

                }
                System.out.println("layer: " + layer + " neuron: " + neuron + ": " + pDerivatives);
                layerDerivatives[layer][neuron] = pDerivatives;
            }
        }
        for (int layer = 0; layer < layerDerivatives.length; layer++) {
            for (int neuron = 0; neuron < layerDerivatives[layer].length; neuron++) {
                System.out.print(layerDerivatives[layer][neuron] + ", ");
            }
            System.out.println();
        }
        System.out.println();
        return layerDerivatives;
    }

    public Double[][] getBiasDerivatives(Vector[] neuronActivations, Vector input) {
        System.out.println("Biases: ");
        Double[][] biasDerivatives = new Double[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            double bderivative;
            if (layer == 0) {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(input));
                    biasDerivatives[layer][neuron] = bderivative;
                }
            } else {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]));
                    biasDerivatives[layer][neuron] = bderivative;
                }
            }
        }
        for (Double[] layer : biasDerivatives) {
            System.out.println(Arrays.toString(layer));
        }
        return biasDerivatives;
    }

    public Vector getDerivativeOfLossWithLastLayer(Vector actual, Vector prediction) { //derivatives calculated for MSE
        Vector out = new Vector(actual.length());
        for (int i = 0; i < out.length(); i++) {
            out.set(i, -(actual.get(i) - prediction.get(i)));
        }
        return out;
    }

    public Vector[] getLossDWRTactivations(Vector lastLayerDerivatives, Vector[][] layerDerivatives) {
        Vector[] activationDerivatives = new Vector[layers];
        activationDerivatives[layers - 1] = lastLayerDerivatives.copy();
        for (int layer = layers - 1; layer >= 1; layer--) {
            Vector pDerivatives = new Vector(neuronsPerLayer);
            for (int neuron2 = 0; neuron2 < neuronsPerLayer; neuron2++) {
                double completeDerivative = 0;
                for (int neuron1 = 0; neuron1 < neuronsPerLayer; neuron1++) {
                    completeDerivative += layerDerivatives[layer][neuron1].get(neuron2) * lastLayerDerivatives.get(neuron1);
                }
                pDerivatives.set(neuron2, completeDerivative);
            }
            lastLayerDerivatives = pDerivatives.copy();
            activationDerivatives[layer - 1] = pDerivatives;
        }
        return activationDerivatives;
    }

    public Vector[][] getWeightDWRTLoss(Vector[] activationDWRTLoss, Vector[][] weightDWRTactivations) {
        Vector[][] weightDWRTLoss = new Vector[layers][neuronsPerLayer];
        for (int layer = 0; layer < layers; layer++) {
            for (int neuron = 0; neuron < neuronsPerLayer; neuron++) {
                Vector weightDerivatives;
                if (layer == 0) {
                    weightDerivatives = new Vector(inputSize);
                    for (int weight = 0; weight < inputSize; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations[layer][neuron].get(weight));
                    }
                } else {
                    weightDerivatives = new Vector(neuronsPerLayer);
                    for (int weight = 0; weight < neuronsPerLayer; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations[layer][neuron].get(weight));
                    }
                    weightDWRTLoss[layer][neuron] = weightDerivatives;
                }
                weightDWRTLoss[layer][neuron] = weightDerivatives;
            }
        }
        return weightDWRTLoss;
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
