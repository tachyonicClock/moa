package moa.evaluation;

import com.yahoo.labs.samoa.instances.Instance;

import moa.core.Example;
import moa.core.Measurement;
import moa.learners.Learner;
import moa.streams.ExampleStream;

import java.util.ArrayList;
import java.util.HashMap;

public class EfficientEvaluationLoops {

    public static class PrequentialResult {
        public ArrayList<double[]> windowedResults;
        public double[] cumulativeResults;
        public ArrayList<Integer> targets;
        public ArrayList<Integer> predictions;
        public HashMap<String, Double> otherMeasurements;

        public PrequentialResult(
                ArrayList<double[]> windowedResults,
                double[] cumulativeResults,
                ArrayList<Integer> targets,
                ArrayList<Integer> predictions,
                HashMap<String, Double> otherMeasurements) {
            this.windowedResults = windowedResults;
            this.cumulativeResults = cumulativeResults;
            this.targets = targets;
            this.predictions = predictions;
            this.otherMeasurements = otherMeasurements;
        }

        public PrequentialResult(
                ArrayList<double[]> windowedResults,
                double[] cumulativeResults,
                ArrayList<Integer> targets,
                ArrayList<Integer> predictions) {
            this(windowedResults, cumulativeResults, targets, predictions, null);
        }

        public PrequentialResult(
                ArrayList<double[]> windowedResults,
                double[] cumulativeResults,
                HashMap<String, Double> otherMeasurements) {
            this(windowedResults, cumulativeResults, null, null, otherMeasurements);
        }
    }

    public static PrequentialResult prequentialEvaluationClassification(
            ExampleStream stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            boolean storeY,
            boolean storePredictions,
            HookableEvaluationLoop loop) {
        if (!stream.getHeader().outputAttribute(1).isNominal()) {
            throw new IllegalArgumentException(
                    "Stream class attribute is not nominal. Use the regression variant instead.");
        }

        ArrayList<Integer> targetValues = new ArrayList<>();
        ArrayList<Integer> predictions = new ArrayList<>();
        ArrayList<double[]> windowedResults = new ArrayList<>();

        loop.onClassifierPrediction(
                (i, instance, votes) -> {
                    if (storeY) targetValues.add((int) Math.round(instance.getData().classValue()));
                    if (storePredictions) predictions.add(moa.core.Utils.maxIndex(votes));
                    if (basicEvaluator != null) basicEvaluator.addResult(instance, votes);
                    if (windowedEvaluator != null) windowedEvaluator.addResult(instance, votes);
                });

        if (windowedEvaluator != null) {
            loop.onClassifierWindow(
                    (u, votes, labels) -> windowedResults.add(flatten(windowedEvaluator)));
        }

        loop.runClassifier(stream, learner, maxInstances, (int) windowSize);

        double[] cumulativeResults = basicEvaluator != null ? flatten(basicEvaluator) : null;
        return new PrequentialResult(windowedResults, cumulativeResults, targetValues, predictions);
    }

    public static PrequentialResult prequentialEvaluationClassification(
            ExampleStream stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            boolean storeY,
            boolean storePredictions) {
        return prequentialEvaluationClassification(
                stream,
                learner,
                basicEvaluator,
                windowedEvaluator,
                maxInstances,
                windowSize,
                storeY,
                storePredictions,
                new HookableEvaluationLoop());
    }

    public static PrequentialResult prequentialEvaluationRegression(
            ExampleStream stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            boolean storeY,
            boolean storePredictions,
            HookableEvaluationLoop loop) {
        if (!stream.getHeader().outputAttribute(1).isNumeric()) {
            throw new IllegalArgumentException(
                    "Stream class attribute is not numeric. Use the classification variant"
                            + " instead.");
        }
        if (storePredictions) {
            throw new IllegalArgumentException(
                    "Cannot store numeric predictions for regression tasks.");
        }
        if (storeY) {
            throw new IllegalArgumentException(
                    "Cannot store numeric target values for regression tasks.");
        }

        // Leave empty for regression tasks to satisfy PrequentialResult signature
        ArrayList<Integer> targetValues = new ArrayList<>();
        ArrayList<Integer> predictions = new ArrayList<>();
        ArrayList<double[]> windowedResults = new ArrayList<>();

        loop.onRegressorPrediction(
                (i, instance, prediction) -> {
                    double[] votes = new double[] {prediction}; // Compatibility hack
                    if (basicEvaluator != null) basicEvaluator.addResult(instance, votes);
                    if (windowedEvaluator != null) windowedEvaluator.addResult(instance, votes);
                });

        if (windowedEvaluator != null) {
            loop.onRegressorWindow(
                    (u, votes, labels) -> windowedResults.add(flatten(windowedEvaluator)));
        }

        loop.runRegressor(stream, learner, maxInstances, (int) windowSize);

        double[] cumulativeResults = basicEvaluator != null ? flatten(basicEvaluator) : null;
        return new PrequentialResult(windowedResults, cumulativeResults, targetValues, predictions);
    }

    public static PrequentialResult prequentialEvaluationRegression(
            ExampleStream<Example<Instance>> stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            boolean storeY,
            boolean storePredictions) {
        return prequentialEvaluationRegression(
                stream,
                learner,
                basicEvaluator,
                windowedEvaluator,
                maxInstances,
                windowSize,
                storeY,
                storePredictions,
                new HookableEvaluationLoop());
    }

    /** The current value of every measurement the evaluator reports, in its own order. */
    private static double[] flatten(LearningPerformanceEvaluator<Example<Instance>> evaluator) {
        Measurement[] measurements = evaluator.getPerformanceMeasurements();
        double[] values = new double[measurements.length];
        for (int i = 0; i < values.length; ++i) values[i] = measurements[i].getValue();
        return values;
    }

    public static PrequentialResult prequentialSSLEvaluation(
            ExampleStream<Example<Instance>> stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            long initialWindowSize,
            long delayLength,
            double labelProbability,
            int randomSeed,
            boolean debugPseudoLabels,
            boolean storeY,
            boolean storePredictions,
            HookableEvaluationLoop loop) {
        if (!stream.getHeader().outputAttribute(1).isNominal()) {
            throw new IllegalArgumentException(
                    "Stream class attribute is not nominal. Use the regression variant instead.");
        }
        // Hacky way to allow modification within lambda expressions
        // TODO: Should probably pop these out into a separate object that we register.
        int[] numUnlabeledData = {0};
        int[] numCorrectPseudoLabeled = {0};
        int[] numInstancesTested = {0};
        HashMap<String, Double> otherMeasures = new HashMap<>();

        ArrayList<Integer> targetValues = new ArrayList<>();
        ArrayList<Integer> predictions = new ArrayList<>();
        ArrayList<double[]> windowedResults = new ArrayList<>();

        loop.onClassifierPrediction(
                (i, instance, votes) -> {
                    numInstancesTested[0]++;
                    if (storeY) targetValues.add((int) Math.round(instance.getData().classValue()));
                    if (storePredictions) predictions.add(moa.core.Utils.maxIndex(votes));
                    if (basicEvaluator != null) basicEvaluator.addResult(instance, votes);
                    if (windowedEvaluator != null) windowedEvaluator.addResult(instance, votes);
                });
        loop.onClassifierPseudoLabel(
                (long i, Instance instance, int pseudoLabel, int trueLabel) -> {
                    numUnlabeledData[0]++;
                    if (pseudoLabel == trueLabel) {
                        numCorrectPseudoLabeled[0]++;
                    }
                });

        if (windowedEvaluator != null) {
            loop.onClassifierWindow(
                    (u, votes, labels) -> windowedResults.add(flatten(windowedEvaluator)));
        }

        loop.runClassifier(stream, learner, maxInstances, (int) windowSize);

        double[] cumulativeResults = basicEvaluator != null ? flatten(basicEvaluator) : null;

        // TODO: Add this measures in a windowed way.
        otherMeasures.put("num_unlabeled_instances", (double) numUnlabeledData[0]);
        otherMeasures.put("num_correct_pseudo_labeled", (double) numCorrectPseudoLabeled[0]);
        otherMeasures.put("num_instances_tested", (double) numInstancesTested[0]);
        otherMeasures.put(
                "pseudo_label_accuracy",
                (double) numCorrectPseudoLabeled[0] / numInstancesTested[0]);

        return new PrequentialResult(
                windowedResults, cumulativeResults, targetValues, predictions, otherMeasures);
    }

    public static PrequentialResult prequentialSSLEvaluation(
            ExampleStream<Example<Instance>> stream,
            Learner learner,
            LearningPerformanceEvaluator<Example<Instance>> basicEvaluator,
            LearningPerformanceEvaluator<Example<Instance>> windowedEvaluator,
            long maxInstances,
            long windowSize,
            long initialWindowSize,
            long delayLength,
            double labelProbability,
            int randomSeed,
            boolean debugPseudoLabels,
            boolean storeY,
            boolean storePredictions) {
        return prequentialSSLEvaluation(
                stream,
                learner,
                basicEvaluator,
                windowedEvaluator,
                maxInstances,
                windowSize,
                initialWindowSize,
                delayLength,
                labelProbability,
                randomSeed,
                debugPseudoLabels,
                storeY,
                storePredictions,
                new HookableEvaluationLoop());
    }
}
