package attention;

import CommunicationInterface.SensorI;
import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import org.cst.cogscore.modules.attention.AttentionData;
import org.cst.cogscore.modules.attention.AttentionEvaluationReport;
import org.cst.cogscore.modules.attention.PosnerAttentionTestRunner;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import java.lang.reflect.Array;
import java.util.Collection;

public class PosnerAttentionCodelet extends Codelet {

    private MemoryObject attentionalMapMO;
    private MemoryObject evaluationResultMO;
    private MemoryObject evaluationSummaryMO;

    private final String experimentId;
    private final boolean autoExperimentFromSimulator;
    private final String architectureName;
    private final SensorI vision;
    private final PosnerAttentionTestRunner.Config config;

    private PosnerAttentionTestRunner runner;
    private int activePosnerExperimentId;

    private int currentEpoch = -1;
    private String lastEvaluatedTrialId = null;

    private final List<AttentionEvaluationReport.TrialResult> currentEpochResults =
            new ArrayList<AttentionEvaluationReport.TrialResult>();

    private AttentionData.TrialInput activeTrialInput = null;
    private String activeTrialKey = null;
    private int activeTrialFrameCounter = 0;

    private long localProcCycle = 0L;
    private Integer latestPhaseCode = null;
    private boolean latestTrialCompleteFlag = false;

    private boolean debug = true;
    private boolean allExperimentsDoneFlushed = false;
    private boolean signalAccessDebugPrinted = false;

    /*
     * Java step-level logging.
     * This CSV is written at every proc() cycle, independently from the
     * per-trial and summary CSV files written by PosnerAttentionTestRunner.
     */
    private boolean stepLoggingEnabled = true;
    private PrintWriter stepLogWriter = null;
    private File stepLogFile = null;

    public PosnerAttentionCodelet(
            String experimentId,
            String architectureName,
            PosnerAttentionTestRunner.Config config,
            SensorI vision
    ) {
        this.experimentId = experimentId == null ? "attention_experiment" : experimentId;
        this.autoExperimentFromSimulator = isAutoExperimentId(this.experimentId);
        this.architectureName = architectureName == null || architectureName.trim().isEmpty()
                ? "unknown"
                : architectureName;
        this.vision = vision;
        this.config = config == null ? new PosnerAttentionTestRunner.Config() : config;

        this.activePosnerExperimentId = parseExperimentId(
                this.experimentId,
                PosnerAttentionTestRunner.EXP1_CENTRAL_CUE
        );

        validateExperimentId(this.activePosnerExperimentId);
        this.runner = new PosnerAttentionTestRunner(this.activePosnerExperimentId, this.config);

        setTimeStep(50);
    }

    public PosnerAttentionCodelet(
            int posnerExperimentId,
            String architectureName,
            PosnerAttentionTestRunner.Config config,
            SensorI vision
    ) {
        validateExperimentId(posnerExperimentId);

        this.experimentId = Integer.toString(posnerExperimentId);
        this.autoExperimentFromSimulator = false;
        this.architectureName = architectureName == null || architectureName.trim().isEmpty()
                ? "unknown"
                : architectureName;
        this.vision = vision;
        this.config = config == null ? new PosnerAttentionTestRunner.Config() : config;

        this.activePosnerExperimentId = posnerExperimentId;
        this.runner = new PosnerAttentionTestRunner(this.activePosnerExperimentId, this.config);

        setTimeStep(50);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public void setStepLoggingEnabled(boolean stepLoggingEnabled) {
        this.stepLoggingEnabled = stepLoggingEnabled;

        if (!stepLoggingEnabled) {
            closeStepLogWriter();
        }
    }

    public File getStepLogFile() {
        return stepLogFile;
    }

    public int getPosnerExperimentId() {
        return activePosnerExperimentId;
    }

    private AttentionData.TrialInput newTrialInput() {
        return new AttentionData.TrialInput();
    }

    private AttentionData.Point newPoint(double x, double y) {
        return new AttentionData.Point(x, y);
    }

    @Override
    public void accessMemoryObjects() {
        attentionalMapMO = firstInput("ATTENTIONAL_MAP");

        evaluationResultMO = firstOutput(
                "ATTENTION_EVALUATION_RESULT",
                "POSNER_EVALUATION_RESULT"
        );

        evaluationSummaryMO = firstOutput(
                "ATTENTION_EVALUATION_SUMMARY",
                "POSNER_EVALUATION_SUMMARY"
        );
    }

    @Override
    public void calculateActivation() {
        /*
         * Esta codelet apenas avalia o benchmark.
         * Não compete por ativação.
         */
    }

    @Override
    public void proc() {
        localProcCycle++;

        try {
            latestPhaseCode = null;
            latestTrialCompleteFlag = false;

            Integer epoch = resolveEpoch();
            if (epoch == null) {
                epoch = Integer.valueOf(0);
            }

            if (currentEpoch < 0) {
                currentEpoch = epoch.intValue();
            } else if (epoch.intValue() != currentEpoch) {
                flushCurrentEpoch(false);
                currentEpoch = epoch.intValue();
                lastEvaluatedTrialId = null;
                activeTrialInput = null;
                activeTrialKey = null;
                activeTrialFrameCounter = 0;
            }

            if (debug) {
                System.out.println("Posner | epoch: " + currentEpoch);
            }

            int experimentForThisCycle = resolveExperimentIdFromSimulator();
            ensureRunnerExperiment(experimentForThisCycle);

            if (attentionalMapMO == null) {
                writeJavaStepCsv("NO_ATTENTIONAL_MAP_MO", null, null, 0);

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] skipped: ATTENTIONAL_MAP input MO not found");
                }
                return;
            }

            
            
            Object attentionalMapValue = attentionalMapMO.getI();

            if (attentionalMapValue == null) {
                writeJavaStepCsv("NULL_ATTENTIONAL_MAP", null, null, 0);

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] waiting: ATTENTIONAL_MAP is null");
                }
                return;
            }
            AttentionData.TrialInput snapshot = readTrialContextFromSimulatorSignals();

            if (snapshot == null) {
                writeJavaStepCsv("NO_POSNER_SIGNALS", null, null, 0);
                flushIfAllExperimentsDone();

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] waiting: ATTENTIONAL_MAP exists, but posner_* simulator signals are not available");
                    debugSignalAccessOnce();
                }

                return;
            }

            if (snapshot.episode == 0) {
                snapshot.episode = currentEpoch;
            }

            AttentionData.TrialInput input;
            int addedFramesThisStep = 0;

            boolean snapshotAlreadyHasFrames =
                    snapshot.frames != null && !snapshot.frames.isEmpty();

            if (snapshotAlreadyHasFrames) {
                input = snapshot;
            } else {
                input = mergeWithAccumulatedTrial(snapshot);

                long frameCycle = resolveDefaultFrameCycle(input);
                addedFramesThisStep = appendFrameValue(input, attentionalMapValue, frameCycle);

                activeTrialFrameCounter += addedFramesThisStep;

                if (debug && addedFramesThisStep == 0) {
                    System.out.println("[PosnerAttentionCodelet] ATTENTIONAL_MAP found, but unsupported value type: "
                            + safeClassName(attentionalMapValue));
                }
            }

            AttentionData.Frame latestFrameForLog = latestFrame(input);

            if (!isTrialReadyForEvaluation(input)) {
                writeJavaStepCsv("WAITING", input, latestFrameForLog, addedFramesThisStep);

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] waiting trial="
                            + input.trialId
                            + " phase="
                            + latestPhaseCode
                            + " frames="
                            + input.frames.size()
                            + " targetOnset="
                            + input.targetOnsetCycle
                            + " detection="
                            + input.externalDetectionCycle
                            + " completeFlag="
                            + latestTrialCompleteFlag);
                }
                return;
            }

            if (!input.isValid()) {
                writeJavaStepCsv("INVALID_TRIAL_INPUT", input, latestFrameForLog, addedFramesThisStep);

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] skipped: incomplete TrialInput " + input.trialId);
                }
                return;
            }

            String trialKey = buildEvaluationKey(input);

            if (trialKey.equals(lastEvaluatedTrialId)) {
                writeJavaStepCsv("DUPLICATE_TRIAL", input, latestFrameForLog, addedFramesThisStep);

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] duplicate trial ignored: " + trialKey);
                }
                return;
            }

            AttentionEvaluationReport.TrialResult result = runner.evaluate(input);
            writeJavaStepCsv("EVALUATED", input, latestFrameForLog, addedFramesThisStep);

            currentEpochResults.add(result);
            lastEvaluatedTrialId = trialKey;
            writeCurrentEpochSnapshot(false);
            if (evaluationResultMO != null) {
                evaluationResultMO.setI(result);
            }

            if (debug) {
                System.out.println("[PosnerAttentionCodelet] evaluated experiment="
                        + activePosnerExperimentId
                        + " epoch="
                        + currentEpoch
                        + " trial="
                        + result.trialId
                        + " type="
                        + result.trialType
                        + " rt="
                        + result.reactionTimeCycles);
            }

            activeTrialInput = null;
            activeTrialKey = null;
            activeTrialFrameCounter = 0;

            flushIfAllExperimentsDone();

        } catch (Exception e) {
            System.err.println("[PosnerAttentionCodelet] proc error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private MemoryObject firstInput(String... names) {
        if (names == null) {
            return null;
        }

        for (String name : names) {
            try {
                MemoryObject mo = (MemoryObject) getInput(name);
                if (mo != null) {
                    return mo;
                }
            } catch (Exception ignored) {
            }
        }

        return null;
    }

    private MemoryObject firstOutput(String... names) {
        if (names == null) {
            return null;
        }

        for (String name : names) {
            try {
                MemoryObject mo = (MemoryObject) getOutput(name);
                if (mo != null) {
                    return mo;
                }
            } catch (Exception ignored) {
            }
        }

        return null;
    }

    private AttentionData.TrialInput readTrialContextFromSimulatorSignals() {
        Integer trialNumber = readIntegerSignal("posner_trial");
        Integer targetOn = readIntegerSignal("posner_target_on");

        latestPhaseCode = readIntegerSignal("posner_phase_code");

        if (debug) {
            System.out.println("DEBUG posner_ready = " + readIntegerSignal("posner_ready"));
            System.out.println("DEBUG posner_phase_code = " + latestPhaseCode);
            System.out.println("DEBUG posner_trial = " + trialNumber);
            System.out.println("DEBUG posner_target_on = " + targetOn);
            System.out.println("DEBUG posner_current_cycle = " + readIntegerSignal("posner_current_cycle"));
            System.out.println("DEBUG posner_target_onset_cycle = " + readIntegerSignal("posner_target_onset_cycle"));
            System.out.println("DEBUG posner_cue_onset_cycle = " + readIntegerSignal("posner_cue_onset_cycle"));
            System.out.println("DEBUG posner_trial_complete = " + readIntegerSignal("posner_trial_complete"));
        }

        if (trialNumber == null && targetOn == null && latestPhaseCode == null) {
            return null;
        }

        AttentionData.TrialInput input = newTrialInput();

        input.episode = intSignal(
                "posner_episode",
                currentEpoch < 0 ? 0 : currentEpoch
        );

        input.trialId = stringSignal(
                "posner_trial_id",
                "E" + activePosnerExperimentId + "_T" + intSignal("posner_trial", 0)
        );

        input.modality = "attention";

        input.cueType = cueTypeFromSignals();
        input.trialType = trialTypeFromSignals();
        input.searchType = searchTypeFromSignals();

        Double targetX = readDoubleSignal("posner_target_x_norm");
        Double targetY = readDoubleSignal("posner_target_y_norm");

        if (targetX != null && targetY != null) {
            input.targetNormalized = newPoint(targetX.doubleValue(), targetY.doubleValue());
        }

        Double cueX = readDoubleSignal("posner_cue_x_norm");
        Double cueY = readDoubleSignal("posner_cue_y_norm");

        if (cueX != null && cueY != null) {
            input.cueNormalized = newPoint(cueX.doubleValue(), cueY.doubleValue());
        }

        Double fixationX = readDoubleSignal("posner_fixation_x_norm");
        Double fixationY = readDoubleSignal("posner_fixation_y_norm");

        if (fixationX != null && fixationY != null) {
            input.fixationNormalized = newPoint(fixationX.doubleValue(), fixationY.doubleValue());
        }

        input.targetRadiusNormalized =
                doubleSignal("posner_target_radius_norm", input.targetRadiusNormalized);

        input.cueOnsetCycle = positiveLongSignal("posner_cue_onset_cycle");
        input.targetOnsetCycle = positiveLongSignal("posner_target_onset_cycle");

        /*
         * Fallback:
         * Sometimes the signal bridge sees phase=target and target_on=1
         * before it successfully reads posner_target_onset_cycle.
         * In that case, infer target onset from current simulator cycle.
         */
        if (input.targetOnsetCycle == null) {
            Long currentCycle = readLongSignal("posner_current_cycle");

            boolean targetPhase =
                    latestPhaseCode != null && latestPhaseCode.intValue() >= 4;

            boolean targetVisible =
                    targetOn != null && targetOn.intValue() != 0;

            if (targetPhase || targetVisible) {
                if (currentCycle != null && currentCycle.longValue() >= 0L) {
                    input.targetOnsetCycle = currentCycle;
                } else {
                    input.targetOnsetCycle = Long.valueOf(localProcCycle);
                }

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] inferred missing targetOnsetCycle="
                            + input.targetOnsetCycle
                            + " phase="
                            + latestPhaseCode
                            + " targetOn="
                            + targetOn);
                }
            }
        }

        Long detectionCycle = positiveLongSignal("posner_detection_cycle");

        if (detectionCycle != null
                && input.targetOnsetCycle != null
                && detectionCycle.longValue() >= input.targetOnsetCycle.longValue()) {
            input.externalDetectionCycle = detectionCycle;
        }

        input.overtMovementCycle = positiveLongSignal("posner_overt_movement_cycle");
        input.soaMs = readDoubleSignal("posner_soa_ms");

        input.overtMovementEnabled =
                booleanFromIntegerSignal("posner_overt_motion_enabled_active");

        Integer distractorCount = readIntegerSignal("posner_distractor_count");
        if (distractorCount != null) {
            input.distractorCount = distractorCount;
        }

        Integer flanked = readIntegerSignal("posner_flanked");
        if (flanked != null) {
            input.flanked = Boolean.valueOf(flanked.intValue() != 0);
        }

        input.flankerDistance = readDoubleSignal("posner_flanker_distance");

        Integer mapWidth = readIntegerSignal("posner_map_width");
        Integer mapHeight = readIntegerSignal("posner_map_height");

        if (mapWidth != null) {
            input.mapWidth = mapWidth.intValue();
        }

        if (mapHeight != null) {
            input.mapHeight = mapHeight.intValue();
        }

        latestTrialCompleteFlag =
                input.externalDetectionCycle != null
                        || Boolean.TRUE.equals(booleanFromIntegerSignal("posner_trial_complete"))
                        || (latestPhaseCode != null
                        && latestPhaseCode.intValue() == 0
                        && input.targetOnsetCycle != null);

        return input;
    }

    private AttentionData.TrialInput mergeWithAccumulatedTrial(AttentionData.TrialInput snapshot) {
        String key = buildAccumulationKey(snapshot);

        if (activeTrialInput == null || activeTrialKey == null || !activeTrialKey.equals(key)) {
            activeTrialInput = snapshot;
            activeTrialKey = key;
            activeTrialFrameCounter = 0;
            return activeTrialInput;
        }

        activeTrialInput.trialId = snapshot.trialId;
        activeTrialInput.episode = snapshot.episode;
        activeTrialInput.modality = snapshot.modality;

        activeTrialInput.cueType = snapshot.cueType;
        activeTrialInput.trialType = snapshot.trialType;
        activeTrialInput.searchType = snapshot.searchType;

        activeTrialInput.targetNormalized = snapshot.targetNormalized;
        activeTrialInput.cueNormalized = snapshot.cueNormalized;
        activeTrialInput.fixationNormalized = snapshot.fixationNormalized;
        activeTrialInput.targetRadiusNormalized = snapshot.targetRadiusNormalized;

        activeTrialInput.cueOnsetCycle = snapshot.cueOnsetCycle;
        activeTrialInput.targetOnsetCycle = snapshot.targetOnsetCycle;
        activeTrialInput.externalDetectionCycle = snapshot.externalDetectionCycle;
        activeTrialInput.overtMovementCycle = snapshot.overtMovementCycle;
        activeTrialInput.soaMs = snapshot.soaMs;

        activeTrialInput.overtMovementEnabled = snapshot.overtMovementEnabled;
        activeTrialInput.distractorCount = snapshot.distractorCount;
        activeTrialInput.flanked = snapshot.flanked;
        activeTrialInput.flankerDistance = snapshot.flankerDistance;

        activeTrialInput.mapWidth = snapshot.mapWidth;
        activeTrialInput.mapHeight = snapshot.mapHeight;

        return activeTrialInput;
    }

    private AttentionData.Frame latestFrame(AttentionData.TrialInput input) {
        if (input == null || input.frames == null || input.frames.isEmpty()) {
            return null;
        }

        return input.frames.get(input.frames.size() - 1);
    }
    
    private boolean allNumbers(Collection<?> values) {
        if (values == null || values.isEmpty()) {
            return false;
        }

        for (Object v : values) {
            if (!(v instanceof Number)) {
                return false;
            }
        }

        return true;
    }

    private boolean allCollectionsOfNumbers(Collection<?> rows) {
        if (rows == null || rows.isEmpty()) {
            return false;
        }

        for (Object row : rows) {
            if (!(row instanceof Collection<?>)) {
                return false;
            }

            Collection<?> rowCollection = (Collection<?>) row;

            if (rowCollection.isEmpty() || !allNumbers(rowCollection)) {
                return false;
            }
        }

        return true;
    }

    private double[][] collectionRowsToMap(List<?> rows) {
        int height = rows.size();
        int width = 0;

        for (Object row : rows) {
            Collection<?> rowCollection = (Collection<?>) row;
            width = Math.max(width, rowCollection.size());
        }

        if (width <= 0) {
            return new double[][]{{0.0}};
        }

        double[][] map = new double[height][width];

        for (int y = 0; y < height; y++) {
            Collection<?> rowCollection = (Collection<?>) rows.get(y);
            int x = 0;

            for (Object raw : rowCollection) {
                if (x >= width) {
                    break;
                }

                map[y][x] = raw instanceof Number ? ((Number) raw).doubleValue() : 0.0;
                x++;
            }
        }

        return map;
    }

    private Object arrayToList(Object array) {
        if (array == null || !array.getClass().isArray()) {
            return array;
        }

        int n = Array.getLength(array);
        List<Object> list = new ArrayList<Object>(n);

        for (int i = 0; i < n; i++) {
            Object item = Array.get(array, i);

            if (item != null && item.getClass().isArray()) {
                list.add(arrayToList(item));
            } else {
                list.add(item);
            }
        }

        return list;
    }

    private void debugUnsupportedAttentionMapValue(Object value) {
        if (value == null) {
            System.out.println("[PosnerAttentionCodelet] ATTENTIONAL_MAP value is null");
            return;
        }

        System.out.println("[PosnerAttentionCodelet] ATTENTIONAL_MAP unsupported content.");
        System.out.println("[PosnerAttentionCodelet] outer class = " + value.getClass().getName());

        if (value instanceof Collection<?>) {
            Collection<?> c = (Collection<?>) value;
            System.out.println("[PosnerAttentionCodelet] collection size = " + c.size());

            int i = 0;
            for (Object item : c) {
                if (i >= 5) {
                    break;
                }

                System.out.println("[PosnerAttentionCodelet] item[" + i + "] class = "
                        + (item == null ? "null" : item.getClass().getName())
                        + ", value = "
                        + String.valueOf(item));

                if (item instanceof Collection<?>) {
                    Collection<?> sub = (Collection<?>) item;
                    System.out.println("[PosnerAttentionCodelet] item[" + i + "] nested size = " + sub.size());

                    int j = 0;
                    for (Object subItem : sub) {
                        if (j >= 5) {
                            break;
                        }

                        System.out.println("[PosnerAttentionCodelet] item[" + i + "][" + j + "] class = "
                                + (subItem == null ? "null" : subItem.getClass().getName())
                                + ", value = "
                                + String.valueOf(subItem));

                        j++;
                    }
                }

                i++;
            }
        }
    }

    private int appendFramesFromCollection(
            AttentionData.TrialInput input,
            Collection<?> collection,
            long firstCycle
    ) {
        if (input == null || collection == null || collection.isEmpty()) {
            return 0;
        }

        List<?> list = new ArrayList<Object>(collection);

        /*
         * Case 1:
         * ATTENTIONAL_MAP is a flat numeric vector:
         * [0.1, 0.2, 0.3, ...]
         */
        if (allNumbers(list)) {
            input.addFrame(firstCycle, vectorToMap(list, input.mapWidth, input.mapHeight));
            return 1;
        }

        /*
        * WinnerPicker stores ATTENTIONAL_MAP as a time window:
        *
        * outer list = time history
        * inner list = 256-dimensional attention vector
        *
        * For the current frame, use only the newest row.
        */
       if (!list.isEmpty() && list.get(list.size() - 1) instanceof Collection<?>) {
           Collection<?> lastRow = (Collection<?>) list.get(list.size() - 1);

           if (!lastRow.isEmpty() && allNumbers(lastRow)) {
               input.addFrame(
                       firstCycle,
                       vectorToMap(
                               new ArrayList<Object>(lastRow),
                               input.mapWidth,
                               input.mapHeight
                       )
               );
               return 1;
           }
       }
        /*
         * Case 2:
         * ATTENTIONAL_MAP is a 2D list:
         * [[0.1, 0.2], [0.3, 0.4], ...]
         */
        if (allCollectionsOfNumbers(list)) {
            input.addFrame(firstCycle, collectionRowsToMap(list));
            return 1;
        }

        /*
         * Case 3:
         * ATTENTIONAL_MAP is a list of frames or arrays.
         */
        int added = 0;
        long cycle = firstCycle;

        for (Object item : list) {
            if (item == null) {
                continue;
            }

            if (item instanceof AttentionData.Frame) {
                input.addFrame((AttentionData.Frame) item);
                added++;
                continue;
            }

            if (item instanceof double[][]) {
                input.addFrame(cycle, (double[][]) item);
                cycle++;
                added++;
                continue;
            }

            if (item instanceof float[][]) {
                input.addFrame(cycle, toDoubleMap((float[][]) item));
                cycle++;
                added++;
                continue;
            }

            if (item instanceof double[]) {
                input.addFrame(cycle, vectorToMap((double[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;
                continue;
            }

            if (item instanceof float[]) {
                input.addFrame(cycle, vectorToMap((float[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;
                continue;
            }

            if (item instanceof int[]) {
                input.addFrame(cycle, vectorToMap((int[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;
                continue;
            }

            if (item instanceof Number[]) {
                input.addFrame(cycle, vectorToMap((Number[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;
                continue;
            }

            if (item instanceof Collection<?>) {
                Collection<?> sub = (Collection<?>) item;

                if (!sub.isEmpty() && allNumbers(sub)) {
                    input.addFrame(cycle, vectorToMap(new ArrayList<Object>(sub), input.mapWidth, input.mapHeight));
                    cycle++;
                    added++;
                }

                continue;
            }

            if (item.getClass().isArray()) {
                Object converted = arrayToList(item);

                if (converted instanceof Collection<?>) {
                    Collection<?> sub = (Collection<?>) converted;

                    if (!sub.isEmpty() && allNumbers(sub)) {
                        input.addFrame(cycle, vectorToMap(new ArrayList<Object>(sub), input.mapWidth, input.mapHeight));
                        cycle++;
                        added++;
                    }
                }
            }
        }

        if (added == 0 && debug) {
            debugUnsupportedAttentionMapValue(collection);
        }

        return added;
    }

    private int appendFrameValue(AttentionData.TrialInput input, Object value, long cycle) {
        if (input == null || value == null) {
            return 0;
        }

        if (value instanceof AttentionData.Frame) {
            input.addFrame((AttentionData.Frame) value);
            return 1;
        }

        if (value instanceof double[][]) {
            input.addFrame(cycle, (double[][]) value);
            return 1;
        }

        if (value instanceof float[][]) {
            input.addFrame(cycle, toDoubleMap((float[][]) value));
            return 1;
        }

        if (value instanceof double[]) {
            input.addFrame(cycle, vectorToMap((double[]) value, input.mapWidth, input.mapHeight));
            return 1;
        }

        if (value instanceof float[]) {
            input.addFrame(cycle, vectorToMap((float[]) value, input.mapWidth, input.mapHeight));
            return 1;
        }

        if (value instanceof int[]) {
            input.addFrame(cycle, vectorToMap((int[]) value, input.mapWidth, input.mapHeight));
            return 1;
        }

        if (value instanceof Number[]) {
            input.addFrame(cycle, vectorToMap((Number[]) value, input.mapWidth, input.mapHeight));
            return 1;
        }

        /*
         * Important for java.util.Collections$SynchronizedRandomAccessList and
         * other synchronized wrappers.
         */
        if (value instanceof Collection<?>) {
            return appendFramesFromCollection(input, (Collection<?>) value, cycle);
        }

        /*
         * Generic fallback for object arrays, primitive arrays, and nested arrays.
         */
        if (value.getClass().isArray()) {
            Object converted = arrayToList(value);
            if (converted instanceof Collection<?>) {
                return appendFramesFromCollection(input, (Collection<?>) converted, cycle);
            }
        }

        return 0;
    }

    private int appendFramesFromList(AttentionData.TrialInput input, List<?> list, long firstCycle) {
        if (input == null || list == null || list.isEmpty()) {
            return 0;
        }

        Object first = list.get(0);

        if (first instanceof Number) {
            double[][] map = vectorToMap(list, input.mapWidth, input.mapHeight);
            input.addFrame(firstCycle, map);
            return 1;
        }

        int added = 0;
        long cycle = firstCycle;

        for (Object item : list) {
            if (item instanceof AttentionData.Frame) {
                input.addFrame((AttentionData.Frame) item);
                added++;

            } else if (item instanceof double[][]) {
                input.addFrame(cycle, (double[][]) item);
                cycle++;
                added++;

            } else if (item instanceof float[][]) {
                input.addFrame(cycle, toDoubleMap((float[][]) item));
                cycle++;
                added++;

            } else if (item instanceof double[]) {
                input.addFrame(cycle, vectorToMap((double[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;

            } else if (item instanceof float[]) {
                input.addFrame(cycle, vectorToMap((float[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;

            } else if (item instanceof int[]) {
                input.addFrame(cycle, vectorToMap((int[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;

            } else if (item instanceof Number[]) {
                input.addFrame(cycle, vectorToMap((Number[]) item, input.mapWidth, input.mapHeight));
                cycle++;
                added++;

            } else if (item instanceof List<?>) {
                List<?> subList = (List<?>) item;

                if (!subList.isEmpty() && subList.get(0) instanceof Number) {
                    input.addFrame(cycle, vectorToMap(subList, input.mapWidth, input.mapHeight));
                    cycle++;
                    added++;
                }
            }
        }

        return added;
    }

    private double[][] vectorToMap(List<?> values, int requestedWidth, int requestedHeight) {
        int n = values.size();
        int width = requestedWidth;
        int height = requestedHeight;

        if (width <= 0 || height <= 0 || width * height != n) {
            int side = (int) Math.round(Math.sqrt((double) n));

            if (side * side == n) {
                width = side;
                height = side;
            } else {
                width = n;
                height = 1;
            }
        }

        double[][] map = new double[height][width];

        for (int i = 0; i < n; i++) {
            Object raw = values.get(i);
            double v = raw instanceof Number ? ((Number) raw).doubleValue() : 0.0;

            int y = i / width;
            int x = i % width;

            if (y < height) {
                map[y][x] = v;
            }
        }

        return map;
    }

    private double[][] vectorToMap(double[] values, int requestedWidth, int requestedHeight) {
        int n = values.length;
        int width = requestedWidth;
        int height = requestedHeight;

        if (width <= 0 || height <= 0 || width * height != n) {
            int side = (int) Math.round(Math.sqrt((double) n));

            if (side * side == n) {
                width = side;
                height = side;
            } else {
                width = n;
                height = 1;
            }
        }

        double[][] map = new double[height][width];

        for (int i = 0; i < n; i++) {
            int y = i / width;
            int x = i % width;

            if (y < height) {
                map[y][x] = values[i];
            }
        }

        return map;
    }

    private double[][] vectorToMap(float[] values, int requestedWidth, int requestedHeight) {
        int n = values.length;
        int width = requestedWidth;
        int height = requestedHeight;

        if (width <= 0 || height <= 0 || width * height != n) {
            int side = (int) Math.round(Math.sqrt((double) n));

            if (side * side == n) {
                width = side;
                height = side;
            } else {
                width = n;
                height = 1;
            }
        }

        double[][] map = new double[height][width];

        for (int i = 0; i < n; i++) {
            int y = i / width;
            int x = i % width;

            if (y < height) {
                map[y][x] = values[i];
            }
        }

        return map;
    }

    private double[][] vectorToMap(int[] values, int requestedWidth, int requestedHeight) {
        int n = values.length;
        int width = requestedWidth;
        int height = requestedHeight;

        if (width <= 0 || height <= 0 || width * height != n) {
            int side = (int) Math.round(Math.sqrt((double) n));

            if (side * side == n) {
                width = side;
                height = side;
            } else {
                width = n;
                height = 1;
            }
        }

        double[][] map = new double[height][width];

        for (int i = 0; i < n; i++) {
            int y = i / width;
            int x = i % width;

            if (y < height) {
                map[y][x] = values[i];
            }
        }

        return map;
    }

    private double[][] vectorToMap(Number[] values, int requestedWidth, int requestedHeight) {
        int n = values.length;
        int width = requestedWidth;
        int height = requestedHeight;

        if (width <= 0 || height <= 0 || width * height != n) {
            int side = (int) Math.round(Math.sqrt((double) n));

            if (side * side == n) {
                width = side;
                height = side;
            } else {
                width = n;
                height = 1;
            }
        }

        double[][] map = new double[height][width];

        for (int i = 0; i < n; i++) {
            int y = i / width;
            int x = i % width;

            if (y < height) {
                map[y][x] = values[i] == null ? 0.0 : values[i].doubleValue();
            }
        }

        return map;
    }

    private double[][] toDoubleMap(float[][] source) {
        double[][] out = new double[source.length][];

        for (int y = 0; y < source.length; y++) {
            out[y] = new double[source[y].length];

            for (int x = 0; x < source[y].length; x++) {
                out[y][x] = source[y][x];
            }
        }

        return out;
    }

    private long resolveDefaultFrameCycle(AttentionData.TrialInput input) {
        Long currentCycle = readLongSignal("posner_current_cycle");

        if (currentCycle != null && currentCycle.longValue() >= 0L) {
            return currentCycle.longValue();
        }

        Long cue = input.cueOnsetCycle;
        Long target = input.targetOnsetCycle;

        long base;

        if (cue != null && cue.longValue() >= 0L) {
            base = cue.longValue();
        } else if (target != null && target.longValue() >= 0L) {
            base = target.longValue();
        } else {
            base = localProcCycle;
        }

        long candidate = base + activeTrialFrameCounter;

        if (latestPhaseCode != null
                && latestPhaseCode.intValue() >= 4
                && target != null
                && target.longValue() >= 0L) {
            candidate = Math.max(candidate, target.longValue() + activeTrialFrameCounter);
        }

        return candidate;
    }

    private boolean isTrialReadyForEvaluation(AttentionData.TrialInput input) {
        if (input == null) {
            return false;
        }

        if (input.frames == null || input.frames.isEmpty()) {
            return false;
        }

        if (input.targetOnsetCycle == null) {
            return false;
        }

        if (input.targetNormalized == null) {
            return false;
        }

        if (input.trialType == null || input.trialType == AttentionData.TrialType.UNDEFINED) {
            return false;
        }

        if (input.externalDetectionCycle != null) {
            return true;
        }

        return latestTrialCompleteFlag;
    }

    private Integer resolveEpoch() {
        try {
            if (vision != null) {
                return vision.getEpoch();
            }
        } catch (Exception e) {
            System.err.println("[PosnerAttentionCodelet] vision.getEpoch() error: " + e.getMessage());
        }

        return null;
    }

    private int resolveExperimentIdFromSimulator() {
        if (!autoExperimentFromSimulator) {
            return activePosnerExperimentId;
        }

        Integer active = readIntegerSignal("posner_exp_id_active");

        if (active != null && isValidExperimentId(active.intValue())) {
            return active.intValue();
        }

        Integer requested = readIntegerSignal("posner_exp_id");

        if (requested != null && isValidExperimentId(requested.intValue())) {
            return requested.intValue();
        }

        return activePosnerExperimentId;
    }

    private void ensureRunnerExperiment(int experimentId) {
        if (!isValidExperimentId(experimentId)) {
            experimentId = PosnerAttentionTestRunner.EXP1_CENTRAL_CUE;
        }

        if (experimentId == activePosnerExperimentId && runner != null) {
            return;
        }

        flushCurrentEpoch(false);

        activePosnerExperimentId = experimentId;
        runner = new PosnerAttentionTestRunner(activePosnerExperimentId, config);

        lastEvaluatedTrialId = null;
        activeTrialInput = null;
        activeTrialKey = null;
        activeTrialFrameCounter = 0;

        if (debug) {
            System.out.println("[PosnerAttentionCodelet] switched to Posner experiment "
                    + activePosnerExperimentId);
        }
    }

    private String buildAccumulationKey(AttentionData.TrialInput input) {
        String id = input.trialId == null || input.trialId.trim().isEmpty()
                ? "trial_ep" + input.episode + "_target_" + input.targetOnsetCycle
                : input.trialId;

        return activePosnerExperimentId + "::" + input.episode + "::" + id;
    }

    private String buildEvaluationKey(AttentionData.TrialInput input) {
        String id = input.trialId == null || input.trialId.trim().isEmpty()
                ? "trial_ep" + input.episode + "_target_" + input.targetOnsetCycle
                : input.trialId;

        return activePosnerExperimentId
                + "::"
                + currentEpoch
                + "::"
                + id
                + "::target_"
                + input.targetOnsetCycle;
    }

    private void writeCurrentEpochSnapshot(boolean aborted) {
        try {
            if (currentEpoch < 0 || currentEpochResults.isEmpty()) {
                return;
            }

            AttentionEvaluationReport.Summary summary = runner.summarize(
                    architectureName,
                    currentEpoch,
                    aborted,
                    new ArrayList<AttentionEvaluationReport.TrialResult>(currentEpochResults)
            );

            runner.writeEpisodeFiles(summary);

            if (evaluationSummaryMO != null) {
                evaluationSummaryMO.setI(summary);
            }

            if (debug) {
                System.out.println("[PosnerAttentionCodelet] snapshot saved after trial. totalTrials="
                        + summary.totalTrials);
            }

        } catch (IOException e) {
            System.err.println("[PosnerAttentionCodelet] writeCurrentEpochSnapshot I/O error: "
                    + e.getMessage());
            e.printStackTrace();

        } catch (Exception e) {
            System.err.println("[PosnerAttentionCodelet] writeCurrentEpochSnapshot error: "
                    + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void flushCurrentEpoch(boolean aborted) {
        writeCurrentEpochSnapshot(aborted);
        currentEpochResults.clear();
    }

    private boolean isAllExperimentsDoneSignal() {
        Boolean done = booleanFromIntegerSignal("posner_all_done");
        return done != null && done.booleanValue();
    }

    private void flushIfAllExperimentsDone() {
        if (isAllExperimentsDoneSignal()) {
            if (!allExperimentsDoneFlushed) {
                flushCurrentEpoch(false);
                allExperimentsDoneFlushed = true;

                if (debug) {
                    System.out.println("[PosnerAttentionCodelet] all experiments done; final summary flushed");
                }
            }
        } else {
            allExperimentsDoneFlushed = false;
        }
    }

    private void writeJavaStepCsv(
            String status,
            AttentionData.TrialInput input,
            AttentionData.Frame frame,
            int addedFramesThisStep
    ) {
        if (!stepLoggingEnabled) {
            return;
        }

        try {
            PrintWriter pw = getStepLogWriter();
            if (pw == null) {
                return;
            }

            Integer simCurrentCycle = readIntegerSignal("posner_current_cycle");
            Integer cueSideSignal = readIntegerSignal("posner_cue_side");
            Integer targetSideSignal = readIntegerSignal("posner_target_side");

            StringBuilder sb = new StringBuilder();

            appendCsv(sb, localProcCycle);
            appendCsv(sb, currentEpoch);
            appendCsv(sb, activePosnerExperimentId);
            appendCsv(sb, status);
            appendCsv(sb, simCurrentCycle);
            appendCsv(sb, cueSideSignal);
            appendCsv(sb, targetSideSignal);

            appendCsv(sb, input == null ? null : input.trialId);
            appendCsv(sb, latestPhaseCode);
            appendCsv(sb, latestTrialCompleteFlag);
            appendCsv(sb, addedFramesThisStep);
            appendCsv(sb, input == null || input.frames == null ? null : input.frames.size());
            appendCsv(sb, frame == null ? null : frame.getCycle());
            appendCsv(sb, frame == null ? null : frame.getWidth());
            appendCsv(sb, frame == null ? null : frame.getHeight());

            appendCsv(sb, input == null || input.cueType == null ? null : input.cueType.name());
            appendCsv(sb, input == null || input.trialType == null ? null : input.trialType.name());
            appendCsv(sb, input == null || input.searchType == null ? null : input.searchType.name());
            appendCsv(sb, input == null ? null : input.distractorCount);
            appendCsv(sb, input == null ? null : input.flanked);
            appendCsv(sb, input == null ? null : input.flankerDistance);
            appendCsv(sb, input == null ? null : input.soaMs);

            appendCsv(sb, input == null ? null : input.cueOnsetCycle);
            appendCsv(sb, input == null ? null : input.targetOnsetCycle);
            appendCsv(sb, input == null ? null : input.externalDetectionCycle);
            appendCsv(sb, input == null ? null : input.overtMovementCycle);

            appendCsv(sb, pointX(input == null ? null : input.targetNormalized));
            appendCsv(sb, pointY(input == null ? null : input.targetNormalized));
            appendCsv(sb, pointX(input == null ? null : input.cueNormalized));
            appendCsv(sb, pointY(input == null ? null : input.cueNormalized));
            appendCsv(sb, pointX(input == null ? null : input.fixationNormalized));
            appendCsv(sb, pointY(input == null ? null : input.fixationNormalized));

            AttentionData.Point peak = frame == null ? null : frame.getPeakNormalized();
            appendCsv(sb, pointX(peak));
            appendCsv(sb, pointY(peak));
            appendCsv(sb, frame == null ? null : frame.getPeakValue());
            appendCsv(sb, frame == null ? null : frame.variance());
            appendCsv(sb, frame == null ? null : frame.normalizedEntropy());

            pw.println(sb.toString());
            pw.flush();

            if (stepLogFile != null) {
                System.out.println("[PosnerAttentionCodelet] Java step saved: "
                        + stepLogFile.getAbsolutePath());
            }
        } catch (Exception e) {
            System.err.println("[PosnerAttentionCodelet] writeJavaStepCsv error: " + e.getMessage());
        }
    }

    private PrintWriter getStepLogWriter() throws IOException {
        if (!stepLoggingEnabled) {
            return null;
        }

        if (stepLogWriter != null) {
            return stepLogWriter;
        }

        File dir = config != null && config.outDir != null
                ? config.outDir
                : new File("attention_posner_out");

        if (!dir.exists()) {
            dir.mkdirs();
        }

        String prefix = config != null && config.filePrefix != null
                ? config.filePrefix
                : "attention_posner";

        stepLogFile = new File(
                dir,
                safeFileName(prefix)
                        + "_java_steps_"
                        + safeFileName(architectureName)
                        + ".csv"
        );

        boolean writeHeader = !stepLogFile.exists() || stepLogFile.length() == 0L;
        stepLogWriter = new PrintWriter(new FileWriter(stepLogFile, true));

        if (writeHeader) {
            stepLogWriter.println(
                    "local_proc_cycle,epoch,active_posner_experiment_id,status,"
                            + "sim_current_cycle,cue_side,target_side,trial_id,phase_code,trial_complete,"
                            + "added_frames_this_step,frame_count,map_frame_cycle,map_width,map_height,"
                            + "cue_type,trial_type,search_type,distractor_count,flanked,flanker_distance,soa_ms,"
                            + "cue_onset_cycle,target_onset_cycle,detection_cycle,overt_movement_cycle,"
                            + "target_x,target_y,cue_x,cue_y,fixation_x,fixation_y,"
                            + "peak_x,peak_y,peak_value,map_variance,normalized_entropy"
            );
            stepLogWriter.flush();
        }

        return stepLogWriter;
    }

    private void closeStepLogWriter() {
        if (stepLogWriter == null) {
            return;
        }

        try {
            stepLogWriter.flush();
            stepLogWriter.close();
        } catch (Exception ignored) {
        } finally {
            stepLogWriter = null;
        }
    }

    private Double pointX(AttentionData.Point p) {
        return p == null ? null : Double.valueOf(p.getX());
    }

    private Double pointY(AttentionData.Point p) {
        return p == null ? null : Double.valueOf(p.getY());
    }

    private void appendCsv(StringBuilder sb, Object value) {
        if (sb.length() > 0) {
            sb.append(',');
        }

        if (value == null) {
            return;
        }

        String s;

        if (value instanceof Double || value instanceof Float) {
            double d = ((Number) value).doubleValue();

            if (Double.isNaN(d) || Double.isInfinite(d)) {
                s = "";
            } else {
                s = String.format(java.util.Locale.US, "%.10f", d);
            }
        } else {
            s = String.valueOf(value);
        }

        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0) {
            sb.append('"').append(s.replace("\"", "\"\"")).append('"');
        } else {
            sb.append(s);
        }
    }

    private String safeFileName(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "unknown";
        }

        return value.trim().replaceAll("[^A-Za-z0-9_.-]", "_");
    }

    @Override
    public void stop() {
        flushCurrentEpoch(true);
        closeStepLogWriter();
        super.stop();
    }

    private boolean isAutoExperimentId(String id) {
        if (id == null) {
            return true;
        }

        String s = id.trim().toLowerCase();

        return s.isEmpty()
                || "attention_experiment".equals(s)
                || "auto".equals(s)
                || "all".equals(s)
                || "all_experiments".equals(s);
    }

    private int parseExperimentId(String id, int fallback) {
        if (id == null) {
            return fallback;
        }

        String s = id.trim().toLowerCase();

        if (s.contains("exp1") || s.contains("central")) {
            return PosnerAttentionTestRunner.EXP1_CENTRAL_CUE;
        }

        if (s.contains("exp2") || s.contains("soa")) {
            return PosnerAttentionTestRunner.EXP2_SOA_SWEEP;
        }

        if (s.contains("exp3") || s.contains("peripheral")) {
            return PosnerAttentionTestRunner.EXP3_PERIPHERAL_CAPTURE;
        }

        if (s.contains("exp4") || s.contains("visual") || s.contains("search")) {
            return PosnerAttentionTestRunner.EXP4_VISUAL_SEARCH;
        }

        if (s.contains("exp5") || s.contains("crowding")) {
            return PosnerAttentionTestRunner.EXP5_CROWDING;
        }

        try {
            int numeric = Integer.parseInt(s);

            if (isValidExperimentId(numeric)) {
                return numeric;
            }

        } catch (NumberFormatException ignored) {
        }

        return fallback;
    }

    private boolean isValidExperimentId(int id) {
        return id >= PosnerAttentionTestRunner.EXP1_CENTRAL_CUE
                && id <= PosnerAttentionTestRunner.EXP5_CROWDING;
    }

    private void validateExperimentId(int id) {
        if (!isValidExperimentId(id)) {
            throw new IllegalArgumentException(
                    "PosnerExperimentId must be between 1 and 5. Received: " + id
            );
        }
    }

    private AttentionData.CueType cueTypeFromSignals() {
        String s = readStringSignal("posner_cue_type");
        Integer code = readIntegerSignal("posner_cue_type_code");

        return cueTypeValue(s, code, AttentionData.CueType.NEUTRAL);
    }

    private AttentionData.TrialType trialTypeFromSignals() {
        String s = readStringSignal("posner_trial_type");
        Integer code = readIntegerSignal("posner_trial_type_code");

        return trialTypeValue(s, code, AttentionData.TrialType.UNDEFINED);
    }

    private AttentionData.SearchType searchTypeFromSignals() {
        String s = readStringSignal("posner_search_type");
        Integer code = readIntegerSignal("posner_search_type_code");

        return searchTypeValue(s, code, AttentionData.SearchType.NONE);
    }

    private AttentionData.CueType cueTypeValue(
            Object raw,
            Integer code,
            AttentionData.CueType fallback
    ) {
        if (raw instanceof AttentionData.CueType) {
            return (AttentionData.CueType) raw;
        }

        if (raw != null) {
            String s = String.valueOf(raw).trim().toUpperCase();

            if ("CENTRAL".equals(s) || "ENDOGENOUS".equals(s)) {
                return AttentionData.CueType.ENDOGENOUS;
            }

            if ("PERIPHERAL".equals(s) || "EXOGENOUS".equals(s)) {
                return AttentionData.CueType.EXOGENOUS;
            }

            if ("NEUTRAL".equals(s)) {
                return AttentionData.CueType.NEUTRAL;
            }

            try {
                return AttentionData.CueType.valueOf(s);
            } catch (Exception ignored) {
            }
        }

        if (code != null) {
            switch (code.intValue()) {
                case 0:
                    return AttentionData.CueType.NEUTRAL;
                case 1:
                    return AttentionData.CueType.ENDOGENOUS;
                case 2:
                    return AttentionData.CueType.EXOGENOUS;
                default:
                    return fallback;
            }
        }

        return fallback;
    }

    private AttentionData.TrialType trialTypeValue(
            Object raw,
            Integer code,
            AttentionData.TrialType fallback
    ) {
        if (raw instanceof AttentionData.TrialType) {
            return (AttentionData.TrialType) raw;
        }

        if (raw != null) {
            try {
                return AttentionData.TrialType.valueOf(
                        String.valueOf(raw).trim().toUpperCase()
                );
            } catch (Exception ignored) {
            }
        }

        if (code != null) {
            switch (code.intValue()) {
                case 1:
                    return AttentionData.TrialType.VALID;
                case 2:
                    return AttentionData.TrialType.INVALID;
                case 3:
                    return AttentionData.TrialType.NEUTRAL;
                default:
                    return fallback;
            }
        }

        return fallback;
    }

    private AttentionData.SearchType searchTypeValue(
            Object raw,
            Integer code,
            AttentionData.SearchType fallback
    ) {
        if (raw instanceof AttentionData.SearchType) {
            return (AttentionData.SearchType) raw;
        }

        if (raw != null) {
            try {
                return AttentionData.SearchType.valueOf(
                        String.valueOf(raw).trim().toUpperCase()
                );
            } catch (Exception ignored) {
            }
        }

        if (code != null) {
            switch (code.intValue()) {
                case 0:
                    return AttentionData.SearchType.NONE;
                case 1:
                    return AttentionData.SearchType.FEATURE;
                case 2:
                    return AttentionData.SearchType.CONJUNCTION;
                default:
                    return fallback;
            }
        }

        return fallback;
    }

    private Integer readIntegerSignal(String signalName) {
        Object value = invokeVisionStringMethod(
                signalName,
                "getIntegerSignal",
                "readIntegerSignal",
                "getIntSignal",
                "readIntSignal"
        );

        if (value instanceof Number) {
            return Integer.valueOf(((Number) value).intValue());
        }

        if (value instanceof String) {
            try {
                return Integer.valueOf(Integer.parseInt(((String) value).trim()));
            } catch (NumberFormatException ignored) {
                return null;
            }
        }

        return null;
    }

    private Long readLongSignal(String signalName) {
        Integer i = readIntegerSignal(signalName);

        if (i != null) {
            return Long.valueOf(i.longValue());
        }

        Double d = readDoubleSignal(signalName);

        if (d != null) {
            return Long.valueOf(d.longValue());
        }

        return null;
    }

    private Long positiveLongSignal(String signalName) {
        Long v = readLongSignal(signalName);

        if (v == null || v.longValue() < 0L) {
            return null;
        }

        return v;
    }

    private Double readDoubleSignal(String signalName) {
        Object value = invokeVisionStringMethod(
                signalName,
                "getFloatSignal",
                "readFloatSignal",
                "getDoubleSignal",
                "readDoubleSignal",
                "getNumberSignal",
                "readNumberSignal"
        );

        if (value instanceof Number) {
            return Double.valueOf(((Number) value).doubleValue());
        }

        if (value instanceof String) {
            try {
                return Double.valueOf(Double.parseDouble(((String) value).trim()));
            } catch (NumberFormatException ignored) {
                return null;
            }
        }

        return null;
    }

    private String readStringSignal(String signalName) {
        Object value = invokeVisionStringMethod(
                signalName,
                "getStringSignal",
                "readStringSignal",
                "getSignal",
                "readSignal"
        );

        return value == null ? null : String.valueOf(value);
    }

    private int intSignal(String signalName, int fallback) {
        Integer v = readIntegerSignal(signalName);
        return v == null ? fallback : v.intValue();
    }

    private double doubleSignal(String signalName, double fallback) {
        Double v = readDoubleSignal(signalName);
        return v == null ? fallback : v.doubleValue();
    }

    private String stringSignal(String signalName, String fallback) {
        String v = readStringSignal(signalName);
        return v == null || v.trim().isEmpty() ? fallback : v;
    }

    private Boolean booleanFromIntegerSignal(String signalName) {
        Integer v = readIntegerSignal(signalName);

        if (v == null) {
            return null;
        }

        return Boolean.valueOf(v.intValue() != 0);
    }

    private Object invokeVisionStringMethod(String signalName, String... methodNames) {
        if (vision == null || methodNames == null) {
            return null;
        }

        Class<?> cls = vision.getClass();

        for (String methodName : methodNames) {
            try {
                Method m = cls.getMethod(methodName, String.class);
                return m.invoke(vision, signalName);
            } catch (Exception ignored) {
            }
        }

        return null;
    }

    private void debugSignalAccessOnce() {
        if (signalAccessDebugPrinted) {
            return;
        }

        signalAccessDebugPrinted = true;

        if (vision == null) {
            System.out.println("[PosnerAttentionCodelet] vision is null; cannot read CoppeliaSim posner_* signals");
            return;
        }

        System.out.println("[PosnerAttentionCodelet] vision class: " + vision.getClass().getName());

        Method[] methods = vision.getClass().getMethods();

        StringBuilder sb = new StringBuilder();
        sb.append("[PosnerAttentionCodelet] available signal-like methods on vision:");

        boolean found = false;

        for (Method m : methods) {
            String name = m.getName().toLowerCase();

            if (name.contains("signal")
                    || name.contains("integer")
                    || name.contains("float")
                    || name.contains("double")
                    || name.contains("string")) {
                sb.append("\n  ").append(m.toString());
                found = true;
            }
        }

        if (found) {
            System.out.println(sb.toString());
        } else {
            System.out.println("[PosnerAttentionCodelet] no public signal-like methods found on vision object");
        }
    }

    private String safeClassName(Object value) {
        if (value == null) {
            return "null";
        }

        try {
            return value.getClass().getName();
        } catch (Exception e) {
            return "unknown";
        }
    }
}