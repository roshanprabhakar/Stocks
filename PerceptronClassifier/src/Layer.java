public class Layer implements NetworkConstants {

    private Perceptron[] neurons;

    public Layer(int n) {
        neurons = new Perceptron[n];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
        }
    }

    public Perceptron get(int i) {
        return neurons[i];
    }

    public Vector activations(Vector previous) { //activations from previous layer
        Vector activations = new Vector(neurons.length);
        for (int i = 0; i < neurons.length; i++) {
            activations.set(i, neurons[i].guess(previous));
        }
        return activations;
    }
}
