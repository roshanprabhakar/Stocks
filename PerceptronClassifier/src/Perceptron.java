import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.JFrame;

import java.util.ArrayList;
import java.util.Collections;

public class Perceptron {

    private int dataLength;

    private Vector weights;
    private double bias;

    private String classifyFor;
    private double learningRate;
    private int power;
    private int epochs;

    private static final int TIME_OUT = 20;

    private static final String[] visualizableDimensions = {"x", "y", "z", "a"};

    public Perceptron(int dataLength, String classifyFor, double learningRate, int epochs, int power) {
        weights = new Vector(dataLength * power);
        bias = 0;
        this.classifyFor = classifyFor;
        this.learningRate = learningRate;
        this.power = power;
        this.epochs = epochs;
        this.dataLength = dataLength;
    }

    //remove automatic epochs, add manual epochs: gives a numeric success measurement
    //train all datesets to be tested with the same number epoch
    public void train(ArrayList<Data> trainingDataSet) {
        for (int rep = 0; rep < epochs; rep++) {
            for (Data trainingData : trainingDataSet) {
                train(trainingData);
            }
        }
    }

    public void visualTrain(ArrayList<Data> trainingDataSet, int numDataSets) throws InterruptedException {
        Graph graph = new Graph(trainingDataSet, numDataSets);
        graph.displayChart();
        for (int rep = 0; rep < epochs; rep++) {
            Collections.shuffle(trainingDataSet);
            for (Data trainingData : trainingDataSet) {
                graph.mark(trainingData.getVector().get(0), trainingData.getVector().get(1));
                train(trainingData);
                graph.drawLine(graph.generateLine(weights, bias));
                Thread.sleep(1000);
                graph.clearLine();
                graph.removeMarker();
            }
        }
    }

    private double train(Data trainingData) {
//        System.out.println();
//        System.out.println("------------------------");
        double guess = guess(trainingData.getVector());
//        System.out.println("training data: " + trainingData.getVector());
//        System.out.println("guess: " + guess);
//        System.out.println("old bias: " + bias);
        int actual = map(trainingData.getId());
//        System.out.println("actual: " + actual);
        double error = actual - guess;
//        System.out.println("error: " + actual + " - " + guess + " = " + error);
//        System.out.println("old weights: " + weights);
        weights = weights.add(trainingData.getVector().expand(power).multiplyScalar(error).multiplyScalar(learningRate));
//        System.out.println("new weights: " + weights);
        bias += error * learningRate;
//        System.out.println("new bias: " + bias);
//        System.out.println("------------------------");
        return error;
    }

    public double test(ArrayList<Data> testData) { //obsolete with non thresholding activation
        double correct = 0;
        for (Data data : testData) {
            double guess = guess(data.getVector()); //requires threshold guessing, currently implemented sigmoid
            if (guess >= 0.5) guess = 1; else if (guess < 0.5) guess = 0;
            int actual = this.map(data.getId());
//            System.out.println("guess: " + guess);
//            System.out.println("actual: " + actual);
            if (guess == actual) {
                correct++;
            }
        }
        return correct / testData.size();
    }

    public double guess(Vector input) {
        double guess = sigmoidActivation(weights.cross(input.expand(power)) + bias);
        return guess;
    }

    public double sigmoidActivation(double guess) {
        return Math.exp(guess) / (Math.exp(guess) + 1);
    }

    public int map(String id) {
        if (id.equals(classifyFor)) return 1;
        return 0;
    }

    public Vector getWeights() {
        return this.weights;
    }

    public double getBias() {
        return bias;
    }

    public String equation() {
        String out = "";
        for (int i = 0; i < dataLength; i++) {
            for (int pow = 1; pow <= power; pow++) {
                System.out.print(weights.get(i * power + pow - 1) + visualizableDimensions[i] + "^" + pow + " + ");
            }
        }
        System.out.println(bias + " = 0");
        return out;
    }
}
