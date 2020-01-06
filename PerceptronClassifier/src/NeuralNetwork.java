import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork implements NetworkConstants { //TODO write output layer calculations with neuron counts different than those of hidden layers

    private ArrayList<Layer> network;
    private int neuronsPerHiddenLayer;
    private int neuronsInOutput;
    private int inputSize; //unused but probably necessary
    private int layers;

    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers, int numOutputNeurons) { //TODO solve output layers
        network = new ArrayList<>();
        for (int i = 0; i < numHiddenLayers; i++) {
            if (i == 0) network.add(new Layer(neuronsPerHidden, inputSize));
            else network.add(new Layer(neuronsPerHidden, neuronsPerHidden));
        }
        network.add(new Layer(numOutputNeurons, neuronsPerHidden));

        this.inputSize = inputSize;
        this.neuronsPerHiddenLayer = neuronsPerHidden;
        this.neuronsInOutput = numOutputNeurons;
        this.layers = network.size();
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

    public void train(ArrayList<NetworkData> data) {
        for (NetworkData networkData : data) {
            train(networkData.getInput(), networkData.getCorrect());
        }
    }

    public void train(Vector input, Vector correct) {

        //Finding derivatives of the loss with the individual weights
        ForwardPropOutput output = forwardProp(input); //WORKS WITH OUTPUT LAYER
        Vector prediction = output.getResultant(); //needed for calculating loss
        Vector[] neuronActivations = output.getIntermediaryMatrix();

        ArrayList<Vector[]> weightsDWRTLayers = getWeightDWRTLayers(neuronActivations, input);
        ArrayList<Double[]> biasDWRTLayers = getBiasDWRTLayers(neuronActivations, input);

        ArrayList<Vector[]> layerDWRTpreviousLayers = getLayerDWRTPreviousLayers(neuronActivations);
        Vector lastLayerDWRTLoss = getLastLayerDWRTLoss(correct, neuronActivations[neuronActivations.length - 1]);

        Vector[] activationsDWRTLoss = getActivationDWRTLoss(lastLayerDWRTLoss, layerDWRTpreviousLayers);

        ArrayList<Vector[]> weightDWRTLoss = getWeightsDWRTLoss(activationsDWRTLoss, weightsDWRTLayers);
        ArrayList<Double[]> biasDWRTLoss = getBiasDWRTLoss(activationsDWRTLoss, biasDWRTLayers);

        //END COMPUTATIONS
//
//        System.out.println("COMPLETE DERIVATIVES OF LOSS WITH RESPECT TO EACH INDIVIDUAL WEIGHT");
//        for (int layer = 0; layer < layers; layer++) {
//            System.out.println("----------------- LAYER: " + layer + " -----------------");
//            for (int neuron = 0; neuron < neuronsPerHiddenLayer; neuron++) {
//                System.out.println("----------------- NEURON: " + neuron + " -----------------");
//                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
//                    System.out.println(weightDWRTLoss.get(layer)[neuron].get(weight));
//                }
//            }
//        }
//        System.out.println();
//
//        System.out.println("COMPLETE DERIVATIVES OF LOSS WITH RESPECT TO EACH INDIVIDUAL BIAS");
//        for (int layer = 0; layer < layers; layer++) {
//            System.out.println("----------------- LAYER: " + layer + " -----------------");
//            for (int neuron = 0; neuron < neuronsPerHiddenLayer; neuron++) {
//                System.out.println("----------------- NEURON: " + neuron + " -----------------");
//                System.out.println(biasDWRTLoss.get(layer)[neuron]);
//            }
//        }
//
//        displayWeightsAndBiases();
//
        //update weights and biases
//        for (int layer = 0; layer < layers; layer++) {
//            for (int neuron = 0; neuron < neuronsPerHiddenLayer; neuron++) {
//                network.get(layer).get(neuron).getWeights().concat(weightDWRTLoss.get(layer)[neuron].multiplyScalar(-1));
//                network.get(layer).get(neuron).updateBias(network.get(layer).get(neuron).getBias() - biasDWRTLoss.get(layer)[neuron]);
//            }
//        }
//
//        displayWeightsAndBiases();
    }

    /**
     * layerDerivatives contains derivative of activation for every neuron with
     * respect to every activation in the previous network layer
     */
    public Vector[] getActivationDWRTLoss(Vector lastLayerDWRTLoss, ArrayList<Vector[]> layerDerivatives) {
        Vector[] activationsDWRTLoss = new Vector[layers];
        activationsDWRTLoss[layers - 1] = lastLayerDWRTLoss;
        for (int layer = layers - 2; layer >= 0; layer--) {
            Vector pDerivatives = new Vector(neuronsPerHiddenLayer);
            for (int neuron2 = 0; neuron2 < neuronsPerHiddenLayer; neuron2++) {
                double derivative = 0;
                for (int neuron1 = 0; neuron1 < activationsDWRTLoss[layer + 1].length(); neuron1++) {
                    derivative += layerDerivatives.get(layer + 1)[neuron1].get(neuron2) * activationsDWRTLoss[layer + 1].get(neuron1);
                }
                pDerivatives.set(neuron2, derivative);
            }
            activationsDWRTLoss[layer] = pDerivatives;
        }
        return activationsDWRTLoss;
    }


    public ArrayList<Vector[]> getWeightDWRTLayers(Vector[] neuronActivations, Vector input) {
        ArrayList<Vector[]> weightDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            System.out.println("----------------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println();
            System.out.println("----------------------------------");
            Vector[] thisLayer = new Vector[network.get(layer).length()];
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                System.out.println("neuron: " + neuron);

                Vector pDerivatives;
                double guess;
                if (layer == 0) {

                    pDerivatives = new Vector(input.length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(input);

                    for (int weight = 0; weight < input.length(); weight++) {
                        System.out.println("weight: " + weight + ": " + network.get(layer).get(neuron).getWeights().get(weight));
                        double factor = input.get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }

                } else {
                    pDerivatives = new Vector(neuronsPerHiddenLayer);

                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        System.out.println("weight: " + weight + ": " + network.get(layer).get(neuron).getWeights().get(weight));
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }
                }

                System.out.println("pDerivatives: " + pDerivatives);
                System.out.println("appending pDerivatives to layer: " + layer + ", neuron: " + neuron);
                thisLayer[neuron] = pDerivatives;
                System.out.println("bias: " + network.get(layer).get(neuron).getBias());
                System.out.println();
            }
            weightDerivatives.add(0, thisLayer);
        }

        //display appropriate derivative information
        for (int layer = 0; layer < weightDerivatives.size(); layer++) {
            for (int neuron = 0; neuron < weightDerivatives.get(layer).length; neuron++) {
                System.out.print(weightDerivatives.get(layer)[neuron] + ",  ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("--------------------------------------------------------------------------------------");
        return weightDerivatives;
    }

    //first null layer in return matrix represents the unusability of the loss derivative with respect to the input vector
    public ArrayList<Vector[]> getLayerDWRTPreviousLayers(Vector[] neuronActivations) {
        System.out.println();
        ArrayList<Vector[]> layerDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            System.out.println();
            System.out.println("----------------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("----------------------------------");
            Vector[] thisLayer = new Vector[network.get(layer).length()];
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(neuronsPerHiddenLayer);
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);

                }
                System.out.println("layer: " + layer + " neuron: " + neuron + ": " + pDerivatives);
                thisLayer[neuron] = pDerivatives;
            }
            layerDerivatives.add(0, thisLayer);
        }
        layerDerivatives.add(0, new Vector[neuronsPerHiddenLayer]);
        for (int layer = 0; layer < layerDerivatives.size(); layer++) {
            for (int neuron = 0; neuron < layerDerivatives.get(layer).length; neuron++) {
                System.out.print(layerDerivatives.get(layer)[neuron] + ", ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("--------------------------------------------------------------------------------------");
        return layerDerivatives;
    }

    public ArrayList<Double[]> getBiasDWRTLayers(Vector[] neuronActivations, Vector input) {
        System.out.println("Biases: ");
        ArrayList<Double[]> biasDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            Double[] bDerivative = new Double[network.get(layer).length()];
            Double bderivative;
            if (layer == 0) {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bDerivative[neuron] = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(input));
                }
            } else {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bDerivative[neuron] = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]));
                }
            }
            biasDerivatives.add(0, bDerivative);
        }
        for (Double[] layer : biasDerivatives) {
            System.out.println(Arrays.toString(layer));
        }
        return biasDerivatives;
    }

    public Vector getLastLayerDWRTLoss(Vector actual, Vector prediction) { //derivatives calculated for MSE

        if (actual.length() != prediction.length()) {
            System.err.print("comparator and prediction not of equal length!");
            System.exit(0);
        }

        Vector out = new Vector(actual.length());
        for (int i = 0; i < out.length(); i++) {
            out.set(i, -(actual.get(i) - prediction.get(i)));
        }
        return out;
    }

    public ArrayList<Vector[]> getWeightsDWRTLoss(Vector[] activationDWRTLoss, ArrayList<Vector[]> weightDWRTactivations) {

        ArrayList<Vector[]> weightDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < layers; layer++) {
            Vector[] thisLayer = new Vector[network.get(layer).length()];
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector weightDerivatives;
                if (layer == 0) {
                    weightDerivatives = new Vector(inputSize);
                    for (int weight = 0; weight < inputSize; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer)[neuron].get(weight));
                    }
                } else {

                    weightDerivatives = new Vector(neuronsPerHiddenLayer);

                    for (int weight = 0; weight < weightDerivatives.length(); weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer)[neuron].get(weight));
                    }
                }
                thisLayer[neuron] = weightDerivatives;
            }
            weightDWRTLoss.add(thisLayer);
        }
        return weightDWRTLoss;
    }

    public ArrayList<Double[]> getBiasDWRTLoss(Vector[] activationDWRTLoss, ArrayList<Double[]> biasDWRTactivations) {
        ArrayList<Double[]> biasDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < layers; layer++) {
            Double[] thisLayer = new Double[network.get(layer).length()];
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                thisLayer[neuron] = biasDWRTactivations.get(layer)[neuron] * activationDWRTLoss[layer].get(neuron);
            }
            biasDWRTLoss.add(thisLayer);
        }
        return biasDWRTLoss;
    }

    public void displayWeightsAndBiases() {
        for (int i = 0; i < 5; i++) System.out.println();
        System.out.println("DISPLAYING WEIGHTS");
        for (int layer = 0; layer < layers; layer++) {
            System.out.println("---------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("---------------------------");
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                System.out.println("NEURON: " + neuron);
                System.out.println("WEIGHTS: " + network.get(layer).get(neuron).getWeights());
                System.out.println("BIAS: " + network.get(layer).get(neuron).getBias());
            }
        }
    }

    public Perceptron getPerceptron(int layer, int perceptron) {
        return network.get(layer).get(perceptron);
    }

    public Layer getLayer(int i) {
        return network.get(i);
    }
}
