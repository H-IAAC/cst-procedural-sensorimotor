
package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.observation.Observation;

import java.util.List;

/**
 * A common interface to all classes capable of handling experience generated by the agents in a learning context.
 *
 * @param <A> Action type
 * @param <E> Experience type
 *
 * @author leolellisr
 */
public interface ExperienceHandler<A, E> {
    void addExperience(Observation observation, A action, double reward, boolean isTerminal);

    /**
     * Called when the episode is done with the last observation
     * @param observation
     */
    void setFinalObservation(Observation observation);

    /**
     * @return The size of the list that will be returned by generateTrainingBatch().
     */
    int getTrainingBatchSize();

    /**
     * The elements are returned in the historical order (i.e. in the order they happened)
     * @return The list of experience elements
     */
    List<E> generateTrainingBatch();

    /**
     * Signal the experience handler that a new episode is starting
     */
    void reset();
}
