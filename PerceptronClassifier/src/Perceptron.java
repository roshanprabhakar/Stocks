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
    public void train(ArrayList<Data> trainingDataSet) throws InterruptedException {
        for (int rep = 0; rep < epochs; rep++) {
            for (Data trainingData : trainingDataSet) {
                train(trainingData);
            }
        }
    }

    public boolean convergeTrain(ArrayList<Data> trainingDataSet, double percentConvergence) throws InterruptedException { //returns true once dataSet is separated
        long start = System.currentTimeMillis();
        double correct = 0;
        while (correct / trainingDataSet.size() < percentConvergence && !timedOut(start, power * 10)) {
            correct = 0;
            for (Data trainingData : trainingDataSet) {
                train(trainingData);
            }
            for (Data trainingData : trainingDataSet) {
                double guess = guess(trainingData.getVector());
                int actual = map(trainingData.getId());
                int error = actual - sigmoidMap(guess);
                if (error == 0) correct++;
            }
//            System.out.println("current: " + correct / trainingDataSet.size());
//            System.out.println("needed: " + percentConvergence);
        }
        if (correct / trainingDataSet.size() >= percentConvergence) return true;
        return false;
    }

    public boolean timedOut(long startTime, int timeOut) {
        if ((System.currentTimeMillis() - startTime) / 1000 >= timeOut) return true;
        else return false;
    }

    private double train(Data trainingData) throws InterruptedException {
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
//        Thread.sleep(10000);
        return error;
    }

    public double test(ArrayList<Data> testData) {
        double correct = 0;
        for (Data data : testData) {
            double guess = sigmoidMap(guess(data.getVector()));
            int actual = this.map(data.getId());
            if (guess == actual) {
                correct++;
            }
        }
        return correct / testData.size();
    }

    public double guess(Vector input) {
        return sigmoidActivation(weights.cross(input.expand(power)) + bias);
    }

    public double sigmoidActivation(double guess) {
        return sigmoid(guess);
    }

    private double sigmoid(double input) { //capped to prevent overflow error, less exponential calculations
        if (input >= 10) {
            return -(1 / (1 + Math.exp(10))) + 1;
        } else if (input <= -10) {
            return -(1 / (1 + Math.exp(-10))) + 1;
        }
        return -(1 / (1 + Math.exp(input))) + 1;
    }

    private int sigmoidMap(double error) {
        if (error >= 0.5) return 1; else return 0;
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

    //VISUALIZATIONS AND TESTING

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

}
