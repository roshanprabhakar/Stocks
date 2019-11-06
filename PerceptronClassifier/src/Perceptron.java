import java.util.ArrayList;

public class Perceptron {

    private Vector weights;
    private String classifyFor;

    private double learningRate;
    private int power;
    private int epochs;

    private static final int TIME_OUT = 20;

    public Perceptron(int dataLength, String classifyFor, double learningRate, int epochs, int power) {
        weights = new Vector(dataLength * power);
        this.classifyFor = classifyFor;
        this.learningRate = learningRate;
        this.power = power;
        this.epochs = epochs;
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

    private int train(Data trainingData) {
        int guess = guess(trainingData.getVector());
        int actual = map(trainingData.getId());
        int error = actual - guess;
        weights = weights.add(trainingData.getVector().expand(power).multiplyScalar(error).multiplyScalar(learningRate));
        return error;
    }

    public double test(ArrayList<Data> testData) {
        double correct = 0;
        for (Data data : testData) {
            int guess = guess(data.getVector());
            int actual = this.map(data.getId());
            if (guess == actual) {correct++;}
        }
        return correct / testData.size();
    }

    public int guess(Vector input) {
        return activation(weights.cross(input.expand(power)));
    }

    public int activation(double guess) {
        if (guess >= 0) {
            return 1;
        } else return 0;
    }

    public int map(String id) {
        if (id.equals(classifyFor)) return 1;
        return 0;
    }

    public Vector getWeights() {
        return this.weights;
    }


//    //remove automatic epochs, add manual epochs: gives a numeric success measurement
//    public void train(ArrayList<Data> trainingDataSet) {
//        double current = 0;
//        long start = System.currentTimeMillis();
//        boolean separable = false;
//        while (current < goal && (separable = !timedOut(start))) {
//            double perfectCount = 0;
//            for (Data trainingData : trainingDataSet) {
//                int error = train(trainingData);
//                if (error == 0) perfectCount++;
//            }
//            current = perfectCount / trainingDataSet.size();
//        }
//        this.separable = separable;
//    }
//
//    private boolean timedOut(long start) {
//        if ((System.currentTimeMillis() - start) / 1000 > TIME_OUT) return true;
//        return false;
//    }

}
