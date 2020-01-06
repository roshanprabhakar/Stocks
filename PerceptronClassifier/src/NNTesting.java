import java.util.Arrays;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0, 3, 2}); //MAKE SURE THIS MATCHES NUMBER OF OUTPUT NEURONS
        Vector input = new Vector(new double[]{1, 2, 3});

        NeuralNetwork network = new NeuralNetwork(input.length(), 2, 2, 3); //num output ignored for now
        network.train(input, actual);

        System.exit(0);

    }
}
