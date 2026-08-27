package moa.evaluation;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.SemiSupervisedLearner;
import moa.core.Example;
import moa.learners.Learner;
import moa.streams.ExampleStream;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Queue;

public class HookableEvaluationLoop {

    public interface Hook {}

    @FunctionalInterface
    public interface OnClassifierPrediction extends Hook {
        void onClassifierPrediction(long i, Example<Instance> instance, double[] votes);
    }

    @FunctionalInterface
    public interface OnClassifierWindow extends Hook {
        void onClassifierWindow(long i, double[][] votes, int[] labels);
    }

    @FunctionalInterface
    public interface OnClassifierPseudoLabel extends Hook {
        void onClassifierPseudoLabel(long i, Instance instance, int pseudoLabel, int trueLabel);
    }

    @FunctionalInterface
    public interface OnRegressorPrediction extends Hook {
        void onRegressorPrediction(long i, Example<Instance> instance, double prediction);
    }

    @FunctionalInterface
    public interface OnRegressorWindow extends Hook {
        void onRegressorWindow(long i, double[] predictions, double[] targets);
    }

    @FunctionalInterface
    public interface OnRegressorPseudoTarget extends Hook {
        void onRegressorPseudoTarget(
                long i, Instance instance, double pseudoTarget, double trueTarget);
    }

    private final List<OnRegressorPrediction> onRegressorPredictionHooks = new ArrayList<>();
    private final List<OnClassifierPrediction> onClassifierPredictionHooks = new ArrayList<>();
    private final List<OnClassifierWindow> onClassifierWindowHooks = new ArrayList<>();
    private final List<OnClassifierPseudoLabel> onClassifierPseudoLabelHooks = new ArrayList<>();
    private final List<OnRegressorWindow> onRegressorWindowHooks = new ArrayList<>();
    private final List<OnRegressorPseudoTarget> onRegressorPseudoTargetHooks = new ArrayList<>();

    // Builders

    public HookableEvaluationLoop onClassifierPrediction(OnClassifierPrediction hook) {
        this.onClassifierPredictionHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop onClassifierWindow(OnClassifierWindow hook) {
        this.onClassifierWindowHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop onClassifierPseudoLabel(OnClassifierPseudoLabel hook) {
        this.onClassifierPseudoLabelHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop onRegressorPrediction(OnRegressorPrediction hook) {
        this.onRegressorPredictionHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop onRegressorWindow(OnRegressorWindow hook) {
        this.onRegressorWindowHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop onRegressorPseudoTarget(OnRegressorPseudoTarget hook) {
        this.onRegressorPseudoTargetHooks.add(Objects.requireNonNull(hook));
        return this;
    }

    public HookableEvaluationLoop register(Hook hook) {
        if (hook instanceof OnRegressorPrediction) {
            this.onRegressorPredictionHooks.add((OnRegressorPrediction) hook);
        }
        if (hook instanceof OnClassifierPrediction) {
            this.onClassifierPredictionHooks.add((OnClassifierPrediction) hook);
        }
        if (hook instanceof OnClassifierWindow) {
            this.onClassifierWindowHooks.add((OnClassifierWindow) hook);
        }
        if (hook instanceof OnClassifierPseudoLabel) {
            this.onClassifierPseudoLabelHooks.add((OnClassifierPseudoLabel) hook);
        }
        if (hook instanceof OnRegressorWindow) {
            this.onRegressorWindowHooks.add((OnRegressorWindow) hook);
        }
        if (hook instanceof OnRegressorPseudoTarget) {
            this.onRegressorPseudoTargetHooks.add((OnRegressorPseudoTarget) hook);
        }
        return this;
    }

    // Notifier helpers

    protected void notifyRegressorPrediction(
            long i, Example<Instance> instance, double prediction) {
        for (OnRegressorPrediction hook : this.onRegressorPredictionHooks)
            hook.onRegressorPrediction(i, instance, prediction);
    }

    protected void notifyClassifierPrediction(long i, Example<Instance> instance, double[] votes) {
        for (OnClassifierPrediction hook : this.onClassifierPredictionHooks)
            hook.onClassifierPrediction(i, instance, votes);
    }

    protected void notifyClassifierWindow(long i, double[][] votes, int[] labels) {
        for (OnClassifierWindow hook : this.onClassifierWindowHooks)
            hook.onClassifierWindow(i, votes, labels);
    }

    protected void notifyClassifierPseudoLabel(
            long i, Instance instance, int pseudoLabel, int trueLabel) {
        for (OnClassifierPseudoLabel hook : this.onClassifierPseudoLabelHooks)
            hook.onClassifierPseudoLabel(i, instance, pseudoLabel, trueLabel);
    }

    protected void notifyRegressorWindow(long i, double[] predictions, double[] targets) {
        for (OnRegressorWindow hook : this.onRegressorWindowHooks)
            hook.onRegressorWindow(i, predictions, targets);
    }

    protected void notifyRegressorPseudoTarget(
            long i, Instance instance, double pseudoTarget, double trueTarget) {
        for (OnRegressorPseudoTarget hook : this.onRegressorPseudoTargetHooks)
            hook.onRegressorPseudoTarget(i, instance, pseudoTarget, trueTarget);
    }

    // Runners

    public void runClassifier(
            ExampleStream<?> stream,
            Learner<Example<Instance>> learner,
            long maxInstances,
            int windowSize) {
        if (!stream.hasMoreInstances()) stream.restart();

        double[][] windowVotes = new double[windowSize][];
        int[] windowLabels = new int[windowSize];

        long i = 0; // Use long to prevent integer overflow on large streams
        for (; stream.hasMoreInstances() && (maxInstances == -1 || i < maxInstances); i++) {
            // Prediction
            Example<Instance> instance = stream.nextInstance();
            double[] votes = learner.getVotesForInstance(instance);

            // Instance Hook
            notifyClassifierPrediction(i, instance, votes);

            // Window Hook
            int windowIndex = (int) (i % windowSize);
            windowVotes[windowIndex] = votes;
            windowLabels[windowIndex] = (int) Math.round(instance.getData().classValue());

            if (windowIndex == windowSize - 1) notifyClassifierWindow(i, windowVotes, windowLabels);

            // Train
            learner.trainOnInstance(instance);
        }

        // Handle last window
        int remainder = (int) (i % windowSize);
        if (remainder > 0) {
            notifyClassifierWindow(
                    i - 1,
                    java.util.Arrays.copyOf(windowVotes, remainder),
                    java.util.Arrays.copyOf(windowLabels, remainder));
        }
    }

    public void runRegressor(
            ExampleStream<?> stream,
            Learner<Example<Instance>> learner,
            long maxInstances,
            int windowSize) {
        if (!stream.hasMoreInstances()) stream.restart();

        double[] windowPredictions = new double[windowSize];
        double[] windowTargets = new double[windowSize];

        long i = 0;
        for (; stream.hasMoreInstances() && (maxInstances == -1 || i < maxInstances); i++) {
            // Prediction
            Example<Instance> instance = stream.nextInstance();
            double prediction = learner.getVotesForInstance(instance)[0];

            // Instance Hook
            notifyRegressorPrediction(i, instance, prediction);

            // Window Hook
            int windowIndex = (int) (i % windowSize);
            windowPredictions[windowIndex] = prediction;
            windowTargets[windowIndex] = instance.getData().classValue();

            if (windowIndex == windowSize - 1)
                notifyRegressorWindow(i, windowPredictions, windowTargets);

            // Train
            learner.trainOnInstance(instance);
        }

        // Handle last window
        int remainder = (int) (i % windowSize);
        if (remainder > 0) {
            notifyRegressorWindow(
                    i - 1,
                    java.util.Arrays.copyOf(windowPredictions, remainder),
                    java.util.Arrays.copyOf(windowTargets, remainder));
        }
    }

    public void runSSLClassifier(
            ExampleStream<Example<Instance>> stream,
            Learner<Example<Instance>> learner,
            long maxInstances,
            int windowSize,
            long initialWindowSize,
            long delayLength,
            double labelProbability,
            int randomSeed) {

        RandomGenerator rng = new MersenneTwister(randomSeed);

        if (!stream.hasMoreInstances()) stream.restart();

        double[][] windowVotes = new double[windowSize][];
        int[] windowLabels = new int[windowSize];

        // Use a Queue to ensure O(1) polling performance for high delay buffers
        Queue<Pair<Long, Example<Instance>>> delayBuffer = new ArrayDeque<>();

        long i = 0;
        long evaluatedCount = 0; // Decoupled from 'i' to handle warmup jumps safely

        for (; stream.hasMoreInstances() && (maxInstances == -1 || i < maxInstances); i++) {

            // Train on delayed instances
            while (!delayBuffer.isEmpty() && delayBuffer.peek().getKey() == i) {
                Example<Instance> delayedExample = delayBuffer.poll().getValue();
                learner.trainOnInstance(delayedExample);
            }

            Example<Instance> instance = stream.nextInstance();
            int trueLabel = (int) Math.round(instance.getData().classValue());

            // Warmup
            if (i < initialWindowSize) {
                learner.trainOnInstance(instance);
                continue; // Skip evaluation hooks during warmup
            }

            // Prediction
            double[] votes = learner.getVotesForInstance(instance);

            // Instance Hook
            notifyClassifierPrediction(i, instance, votes);

            // Window Hook
            int windowIndex = (int) (evaluatedCount % windowSize);
            windowVotes[windowIndex] = votes;
            windowLabels[windowIndex] = trueLabel;

            if (windowIndex == windowSize - 1) notifyClassifierWindow(i, windowVotes, windowLabels);

            evaluatedCount++;

            // Train (SSL & Delay Logic)
            boolean isLabeled = labelProbability > rng.nextDouble();
            int pseudoLabel = -1;

            if (isLabeled && delayLength < 0) {
                learner.trainOnInstance(instance);
            } else {
                // Prepare Unlabeled Example
                Example<Instance> unlabeledInstance = instance.copy();
                // Mask the class value to simulate an unlabeled instance
                Instance __instance = unlabeledInstance.getData();
                __instance.setMissing(__instance.classIndex());

                if (learner instanceof SemiSupervisedLearner) {
                    SemiSupervisedLearner semiLearner = (SemiSupervisedLearner) learner;
                    pseudoLabel = semiLearner.trainOnUnlabeledInstance(unlabeledInstance.getData());
                }
                if (isLabeled) {
                    delayBuffer.add(new MutablePair<>(1 + i + delayLength, instance));
                }
                notifyClassifierPseudoLabel(i, instance.getData(), pseudoLabel, trueLabel);
            }
        }

        // Handle last window
        int remainder = (int) (evaluatedCount % windowSize);
        if (remainder > 0) {
            notifyClassifierWindow(
                    i - 1, // Pass the global stream index where the window closed
                    java.util.Arrays.copyOf(windowVotes, remainder),
                    java.util.Arrays.copyOf(windowLabels, remainder));
        }
    }
}
