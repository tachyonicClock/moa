package moa.evaluation;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.SemiSupervisedLearner;
import moa.core.Example;
import moa.evaluation.HookableEvaluationLoop.Hook;
import moa.learners.Learner;
import moa.streams.ExampleStream;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * A prequential evaluation loop that callers can observe by registering hooks.
 *
 * <p>This class only runs the loop and fires hooks; it has no notion of evaluators or results.
 * {@link EfficientEvaluationLoops#PrequentialEvaluation} is the example caller.
 *
 * <p>Windows are delivered via a fixed-size buffer that is reused (overwritten) every window, so a
 * hook must not retain the {@code votes}/{@code labels} arrays past the call. A trailing partial
 * window, when the stream doesn't divide evenly by the window size, is dropped.
 *
 * <pre>
 * new HookableEvaluationLoop()
 *         .register((OnWindowClose) (votes, labels) -> ...)
 *         .run(stream, learner, 1000, 100);
 * </pre>
 */
public class HookableEvaluationLoop {

    /**
     * Marker for anything that can be passed to {@link #register(Hook)}. Implement one or both of
     * {@link OnLabel} and {@link OnWindowClose}; a hook that implements neither is rejected.
     */
    public interface Hook {
    }

    /** Called once per instance, after the learner has predicted but before it has trained. */
    @FunctionalInterface
    public interface OnLabel extends Hook {
        /**
         * @param i        the index of the instance in the stream.
         * @param instance the instance just tested on. It belongs to the stream and may be reused
         *                 or mutated once this call returns, so a hook must not retain it.
         * @param votes    the learner's raw votes for {@code instance}, as returned by
         *                 {@link Learner#getVotesForInstance(Example)}. All zeros if the learner
         *                 produced no votes at all, which happens while it is still untrained.
         */
        void onLabel(int i, Example<Instance> instance, double[] votes);
    }

    /** Called once per full window of instances. Trailing partial windows are dropped. */
    @FunctionalInterface
    public interface OnWindowClose extends Hook {
        /**
         * @param i      the index of the windows end in the stream.
         * @param votes  one row per instance in the window, each the learner's raw votes for that
         *               instance. This is a reused buffer: a hook must not retain it past the call.
         * @param labels the true class index of each instance in the window, parallel to
         *               {@code votes}. Also a reused buffer.
         */
        void onWindowClose(int i, double[][] votes, int[] labels);
    }

    /** Called once per instance, after the learner has predicted but before it has trained. */
    @FunctionalInterface
    public interface OnPseudoLabel extends Hook {
        /**
         * @param i           the index of the instance in the stream.
         * @param instance    the unlabeled instance just trained on.
         * @param label       the true label of the instance.
         * @param pseudoLabel the pseudo label assigned to the instance.
         */
        void onLabel(int i, Instance instance, int pseudoLabel, int label);
    }

    protected final List<OnLabel> onLabelHooks = new ArrayList<>();
    protected final List<OnWindowClose> onWindowCloseHooks = new ArrayList<>();
    protected final List<OnPseudoLabel> onPseudoLabelHooks = new ArrayList<>();

    /**
     * Registers a callback. A hook implementing both sub-interfaces is registered for both.
     *
     * @throws IllegalArgumentException if the hook implements neither sub-interface, which would
     *                                  otherwise silently do nothing.
     */
    public HookableEvaluationLoop register(Hook hook) {
        boolean registered = false;
        if (hook instanceof OnLabel) {
            this.onLabelHooks.add((OnLabel) hook);
            registered = true;
        }
        if (hook instanceof OnWindowClose) {
            this.onWindowCloseHooks.add((OnWindowClose) hook);
            registered = true;
        }
        if (hook instanceof OnPseudoLabel) {
            this.onPseudoLabelHooks.add((OnPseudoLabel) hook);
            registered = true;
        }
        if (!registered)
            throw new IllegalArgumentException(
                    "Hook must implement OnLabel and/or OnWindowClose, got: " + hook.getClass().getName());
        return this;
    }

    /**
     * Runs test-then-train over the stream, notifying any registered hooks as it goes.
     *
     * @param maxInstances stop after this many instances, or -1 for the whole stream.
     * @param windowSize   how many instances make up a window.
     */
    public void run(ExampleStream<?> stream, Learner<Example<Instance>> learner, long maxInstances, int windowSize) {
        if (!stream.hasMoreInstances())
            stream.restart();

        double[][] windowVotes = new double[windowSize][];
        int[] windowLabels = new int[windowSize];

        int i = 0; // Move index out of loop scope. Used by last window.
        for (; stream.hasMoreInstances() && (maxInstances == -1 || i < maxInstances); i++) {
            // Prediction
            Example<Instance> instance = stream.nextInstance();
            double[] votes = learner.getVotesForInstance(instance);

            // Instance Hook
            for (OnLabel hook : this.onLabelHooks)
                hook.onLabel(i, instance, votes);

            // Window Hook
            int windowIndex = i % windowSize;
            windowVotes[windowIndex] = votes;
            windowLabels[windowIndex] = (int) Math.round(instance.getData().classValue());
            if (windowIndex == windowSize - 1)
                for (OnWindowClose hook : this.onWindowCloseHooks)
                    hook.onWindowClose(i, windowVotes, windowLabels);
            
            // Train
            learner.trainOnInstance(instance);
        }
        
        // Handle last window
        int remainder = i % windowSize;
        if (remainder > 0) {
            for (OnWindowClose hook : this.onWindowCloseHooks)
                hook.onWindowClose(
                    i - 1,
                    java.util.Arrays.copyOf(windowVotes, remainder),
                    java.util.Arrays.copyOf(windowLabels, remainder)
                );
        }
    }

    public void runSSL(
        ExampleStream<Example<Instance>> stream,
        Learner<Example<Instance>> learner,
        long maxInstances,
        int windowSize,
        long initialWindowSize,
        long delayLength,
        double labelProbability,
        int randomSeed,
        boolean debugPseudoLabels) {

        RandomGenerator taskRandom = new MersenneTwister(randomSeed);

        if (!stream.hasMoreInstances())
            stream.restart();

        double[][] windowVotes = new double[windowSize][];
        int[] windowLabels = new int[windowSize];
        List<Pair<Long, Example<Instance>>> delayBuffer = new ArrayList<>();

        int i = 0; // Move index out of loop scope. Used by last window.
        for (; stream.hasMoreInstances() && (maxInstances == -1 || i < maxInstances); i++) {

            // Train on delayed instances
            while (!delayBuffer.isEmpty() && delayBuffer.get(0).getKey() == i) {
                Example<Instance> delayedExample = delayBuffer.remove(0).getValue();
                learner.trainOnInstance(delayedExample);
            }

            Example<Instance> instance = stream.nextInstance();
            int trueLabel = (int) Math.round(instance.getData().classValue());
            
            // Warmup
            if (i < initialWindowSize) {
                learner.trainOnInstance(instance);
                continue; // Skip evaluation hooks during warmup
            }
            
            // Prepare Unlabeled Example
            Example<Instance> unlabeledInstance = instance.copy();
            if (!debugPseudoLabels) {
                Instance __instance = unlabeledInstance.getData();
                __instance.setMissing(__instance.classIndex());
            }

            // Prediction / Test
            double[] votes = learner.getVotesForInstance(unlabeledInstance);

            // Instance Hook
            for (OnLabel hook : this.onLabelHooks)
                hook.onLabel(i, instance, votes);

            // Window Hook
            int windowIndex = i % windowSize;
            windowVotes[windowIndex] = votes;
            windowLabels[windowIndex] = trueLabel;
            if (windowIndex == windowSize - 1) {
                for (OnWindowClose hook : this.onWindowCloseHooks)
                    hook.onWindowClose(i, windowVotes, windowLabels);
            }

            // Train (SSL & Delay Logic)
            boolean isLabeled = labelProbability > taskRandom.nextDouble();
            int pseudoLabel = -1;

            if (isLabeled && delayLength < 0) {
                learner.trainOnInstance(instance);
            } else {
                if (learner instanceof SemiSupervisedLearner) {
                    SemiSupervisedLearner semiLearner = (SemiSupervisedLearner) learner;
                    pseudoLabel = semiLearner.trainOnUnlabeledInstance(unlabeledInstance.getData());
                }
                if (isLabeled) {
                    delayBuffer.add(new MutablePair<>((long) (1 + i + delayLength), instance));
                }
            }

            for (OnPseudoLabel hook : this.onPseudoLabelHooks)
                hook.onLabel(i, instance.getData(), pseudoLabel, trueLabel);
        }

        // Handle last window
        int remainder = i % windowSize;
        if (remainder > 0) {
            for (OnWindowClose hook : this.onWindowCloseHooks)
                hook.onWindowClose(
                    i - 1,
                    java.util.Arrays.copyOf(windowVotes, remainder),
                    java.util.Arrays.copyOf(windowLabels, remainder)
                );
        }
    }
}
