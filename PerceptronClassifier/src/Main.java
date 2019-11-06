import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.annotation.Target;
import java.util.ArrayList;

public class Main {

    private static final double LEARNING_RATE = 0.001;
    private static final int EPOCHS = 30000;
    private static final int POWER = 4;

    private static final String TARGET = "Iris-setosa";

    /**
     * Implementation of the perceptron algorithm for making market decisions based on stock data, currently being developed with the Iris DataSet
     */
    public static void main(String[] args) {
        DataSet set = new DataSet("TestData.txt");
        ArrayList<ArrayList<Data>> dataSets = set.getDataSets(); //all sets of length r within length n data where r < n

        double bestSuccess = 0.0;
        ArrayList<Integer> attributes = new ArrayList<>();
        ArrayList<Data> bestDataSet = new ArrayList<>();
        for (int i = 0; i < dataSets.size(); i++) {
            ArrayList<Data> dataSet = dataSets.get(i);
            System.out.println(set.getPossibleCombinations().get(i));
            Perceptron perceptron = new Perceptron(dataSet.get(0).size(), TARGET, LEARNING_RATE, EPOCHS, POWER);
            perceptron.train(dataSet);
            double success = perceptron.test(dataSet);
            if (success > bestSuccess) {
                attributes = set.getPossibleCombinations().get(i);
                bestDataSet = dataSet;
                bestSuccess = success;
            }
        }

        System.out.println("best attributes: ");
        System.out.println(attributes);

        Perceptron perceptron = new Perceptron(attributes.size(), TARGET, LEARNING_RATE, EPOCHS, POWER);
        perceptron.train(bestDataSet);

        System.out.println("success: ");
        System.out.println(perceptron.test(bestDataSet));
    }

    public static void printLines(int max) {
        for (int i = 0; i < max; i++) System.out.println();
    }
}
