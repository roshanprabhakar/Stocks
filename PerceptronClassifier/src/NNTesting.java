import java.util.Arrays;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0, 2});
        Vector input = new Vector(new double[]{1, 2, 3});

        NeuralNetwork network = new NeuralNetwork(DATA_LENGTH, 2, 3, 1); //num output ignored for now
        network.train(input, actual);

        System.exit(0);

    }
}
