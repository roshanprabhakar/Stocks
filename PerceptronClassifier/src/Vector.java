import java.util.Arrays;
import java.util.function.DoubleToIntFunction;

public class Vector {

    private double[] vector;

    public Vector(double[] vector) {
        this.vector = vector;
    }

    public Vector(int i) {
        this.vector = new double[i];
    }


    public int size() {
        return vector.length;
    }

    public double get(int i) {
        return vector[i];
    }

    public void set(int index, double value) {
        vector[index] = value;
    }


    public Vector multiplyScalar(double scalar) {
        Vector copy = this.copy();
        return multiplyScalar(scalar, copy);
    }

    public static Vector multiplyScalar(double scalar, Vector v) {
        for (int i = 0; i < v.size(); i++) {
            v.set(i, v.get(i) * scalar);
        }
        return v;
    }

    public Vector add(Vector other) {
        Vector copy = this.copy();
        return add(copy, other);
    }

    public static Vector add(Vector v, Vector p) {
        for (int i = 0; i < v.size(); i++) {
            v.set(i, v.get(i) + p.get(i));
        }
        return v;
    }

    public double cross(Vector other) {
        return cross(this, other);
    }

    public static double cross(Vector v, Vector p) {
        double sum = 0;
        String out = "";
        for (int i = 0; i < v.size(); i++) {
            out += v.get(i) + " * " + p.get(i);
            if (i != v.size() - 1) out += " + ";
            sum += v.get(i) * p.get(i);
        }
//        System.out.println(out + " = " + sum);
        return sum;
    }

    public Vector expand(int power) {
        Vector out = new Vector(this.size() * power);
        for (int i = 0; i < size(); i++) {
            for (int pow = 1; pow <= power; pow++) {
                out.set(i * power + pow - 1, Math.pow(this.get(i), pow));
            }
        }

        return out;
    }

//    public double cross(Vector other, int degree) {
//        return cross(this, other, degree);
//    }
//
//    public static double cross(Vector weights, Vector test, int degree) {
//        double sum = 0;
//        for (int i = 0; i < weights.size(); i++) {
//            for (int pow = 1; pow <= degree; pow++) {
//                sum += Math.pow(weights.get(i), pow) * test.get(i);
//            }
//        }
//        return sum;
//    }

    public Vector copy() {
        return new Vector(vector.clone());
    }

    public double[] getVector() {
        return vector;
    }

//    public String toString() {
//        return Arrays.toString(vector);
//    }

    public String toString() {
        String out = "(";
        for (int i = 0; i < vector.length; i++) {
            out += vector[i];
            if (i != vector.length - 1) out += ", ";
        }
        out += ")";
        return out;
    }
}
