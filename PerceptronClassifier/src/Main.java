import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.annotation.Target;
import java.util.ArrayList;

public class Main {

    private static final double LEARNING_RATE = 0.1;
    private static final int EPOCHS = 30;
    private static final int POWER = 1;

    private static final String TARGET = "Iris-setosa";
    private static final int DATA_LENGTH = 2;

    //For visualization purposes
    private static final int NUM_DATA_GROUPS = 3;

    /**
     * Implementation of the perceptron algorithm for making market decisions based on stock data, currently being developed with the Iris DataSet
     */
    public static void main(String[] args) throws InterruptedException {

        DataSet set = new DataSet("TestData.txt");
        ArrayList<Data> data = set.getData(DATA_LENGTH, 0);
//        Graph graph = new Graph(data, NUM_DATA_GROUPS);

        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
//        perceptron.train(data);

        perceptron.visualTrain(data, NUM_DATA_GROUPS);

//        Graph graph = new Graph(data, 3);
//        graph.drawLine(graph.generateLine(perceptron.getWeights(), perceptron.getBias()));
//        graph.displayChart();

        Thread.sleep(100000);
        System.exit(0);

//        DataSet set = new DataSet("TestData.txt");
//        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
//        perceptron.train(set.getData(2, 0));
//        System.out.println("--------------------");
//        System.out.println("DATA: ");
//        DataSet.printSet(set.getData(2, 0));
//        System.out.println("PERCEPTRON: ");
//        System.out.print("Success: ");
//        System.out.println(perceptron.test(set.getData(2, 0)));
//        System.out.println("equation: ");
//        System.out.println(perceptron.equation());
    }

    public static void findBestAttributes(String filepath) { //filepath is the data file !!!BROKEN WITH SIGMOID SUCCESS CALCULATOR!!!
        DataSet set = new DataSet(filepath);
        ArrayList<ArrayList<Data>> dataSets = set.getDataSets(); //all sets of length r within length n data where r < n

        double bestSuccess = 0.0;
        ArrayList<Integer> attributes = new ArrayList<>();
        ArrayList<Data> bestDataSet = new ArrayList<>();
        for (int i = 0; i < dataSets.size(); i++) {
            ArrayList<Data> dataSet = dataSets.get(i);
            System.out.println((int) ((((double) i + 1) / dataSets.size()) * 100) + "% completed");
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
}
