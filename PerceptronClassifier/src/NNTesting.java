import java.util.Arrays;

public class NNTesting {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0, 2});
        Vector input = new Vector(new double[]{1, 2, 3, 4, 5});

        NeuralNetwork network = new NeuralNetwork(NetworkConstants.DATA_LENGTH, 2, 3, 1);
        network.train(input, actual);

        System.exit(0);

    }
}
