package codelets.sensors;

import CommunicationInterface.SensorI;
import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import org.modules.sensorial.CueGenerator;
import org.modules.sensorial.FidelityMetric;
import org.modules.sensorial.DecayModelFitter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import javax.imageio.ImageIO;

/**
 * Sperling-inspired benchmark for Vision integrated as a CST Codelet.
 *
 * What it measures (publishable):
 * - Partial-report fidelity of the agent's stored visual buffer vs. an external reference at t0
 * - Fidelity decay curve across delays
 * - Exponential decay parameter lambda (via DecayModelFitter)
 * - Sensitivity to active interference during retention
 *
 * Data format:
 * - VisionVrep.getData(): List<Float> linear RGB, size = res*res*3, values 0..255
 * - visionBufferMO.getI(): List<Float> representing what the architecture stored (same format)
 */
public class RGBSperling extends Codelet {

    private SensorI visionSensor;           // external reference (proxy GT)
    private MemoryObject visionBufferMO;    // internal buffer under test (List<Float>)
    private final int res;
    private final int patchSize;
    private final long[] delaysMs;
    private final int trialsPerDelay;
    private final boolean activeInterference;
    private final int interferenceReadsPerTick;
    private final long freshFrameTimeoutMs;
    private final File outDir;
    private final Random rnd;
    private final Hooks hooks;

    // Novo: liga/desliga a inserção de ruído no buffer
    private final boolean enableBufferNoise;

    // Parâmetros do ruído no buffer
    private static final float BUFFER_GAUSSIAN_STD = 128.0f;
    private static final boolean APPLY_NOISE_AT_CAPTURE = true;
    private static final boolean APPLY_NOISE_DURING_RETENTION = true;
    private static final int NOISE_WRITES_PER_TICK = 16;

    // Cue generator and metric specialized for List<Float> linear RGB
    private final CueGenerator.Generator<List<Float>> cueGen;
    private final FidelityMetric.Metric<List<Float>> metric;

    // State
    private Phase phase = Phase.WAIT_EPISODE_START;
    private int currentEpisode = -1;

    private int delayIdx = 0;
    private int trialIdx = 0;

    private int warmupFreshFrames = 0;

    private long retentionEndMs = 0;

    // Cached trial data
    private List<Float> groundTruthT0 = null;
    private CueGenerator.Cue<List<Float>> cue = null;

    // Stats per delay for the current episode
    private final double[] sumF;
    private final double[] sumF2;
    private final int[] count;

    // Output writers per episode
    private PrintWriter perTrialWriter = null;

    @Override
    public void accessMemoryObjects() {
        visionBufferMO = (MemoryObject) this.getInput("VISION");
    }

    @Override
    public void calculateActivation() {
    }

    private enum Phase {
        WAIT_EPISODE_START,
        WARMUP,
        CAPTURE_T0,
        RETENTION,
        QUERY_AND_LOG,
        FINISH_EPISODE
    }

    /** Optional hooks to control S0->S1 changes and per-trial resets (recommended for stronger methodology). */
    public interface Hooks {
        void onEpisodeStart(int episode);
        void beforeStimulus(int episode, long delayMs, int trialIdx);
        void afterStimulus(int episode, long delayMs, int trialIdx);
        void duringRetentionTick(int episode, long delayMs, int trialIdx);
        void afterTrial(int episode, long delayMs, int trialIdx);
        void onEpisodeEnd(int episode);
    }

    private static Hooks noopHooks() {
        return new Hooks() {
            public void onEpisodeStart(int episode) {}
            public void beforeStimulus(int episode, long delayMs, int trialIdx) {}
            public void afterStimulus(int episode, long delayMs, int trialIdx) {}
            public void duringRetentionTick(int episode, long delayMs, int trialIdx) {}
            public void afterTrial(int episode, long delayMs, int trialIdx) {}
            public void onEpisodeEnd(int episode) {}
        };
    }

    public RGBSperling(
            SensorI visionSensor,
            int res,
            int patchSize,
            long[] delaysMs,
            int trialsPerDelay,
            boolean activeInterference,
            int interferenceReadsPerTick,
            long freshFrameTimeoutMs,
            File outDir,
            long seed,
            Hooks hooks
    ) {
        this.visionSensor = Objects.requireNonNull(visionSensor, "visionSensor");
        this.res = res;
        this.patchSize = patchSize;
        this.delaysMs = Arrays.copyOf(delaysMs, delaysMs.length);
        this.trialsPerDelay = trialsPerDelay;
        this.activeInterference = activeInterference;
        this.interferenceReadsPerTick = Math.max(0, interferenceReadsPerTick);
        this.freshFrameTimeoutMs = Math.max(50, freshFrameTimeoutMs);
        this.outDir = (outDir == null) ? new File("vision_sperling_out") : outDir;
        this.rnd = new Random(seed);
        this.hooks = (hooks == null) ? noopHooks() : hooks;
        this.enableBufferNoise = true;

        if (!this.outDir.exists()) this.outDir.mkdirs();

        this.sumF = new double[delaysMs.length];
        this.sumF2 = new double[delaysMs.length];
        this.count = new int[delaysMs.length];

        // Cue generator: random patch selection for linear RGB List<Float>
        this.cueGen = new CueGenerator.Generator<List<Float>>() {
            @Override
            public CueGenerator.Cue<List<Float>> sample(Random rnd, List<Float> reference) {
                int x0 = rnd.nextInt(res - patchSize + 1);
                int y0 = rnd.nextInt(res - patchSize + 1);

                return new CueGenerator.Cue<List<Float>>() {
                    @Override
                    public List<Float> extract(List<Float> full) {
                        ArrayList<Float> patch = new ArrayList<>(patchSize * patchSize * 3);
                        for (int y = y0; y < y0 + patchSize; y++) {
                            for (int x = x0; x < x0 + patchSize; x++) {
                                int idx = (y * res + x) * 3;
                                patch.add(full.get(idx));
                                patch.add(full.get(idx + 1));
                                patch.add(full.get(idx + 2));
                            }
                        }
                        return patch;
                    }

                    @Override
                    public String describe() {
                        return "VisionPatchList[x0=" + x0 + ",y0=" + y0 + ",s=" + patchSize + "]";
                    }
                };
            }

            @Override
            public String name() {
                return "VisionPatchList(" + patchSize + ")";
            }
        };

        // Metric: MSE with maxDistance = 255^2 (for normalization inside Sperling runner logic)
        this.metric = new FidelityMetric.Metric<List<Float>>() {
            @Override
            public double distance(List<Float> a, List<Float> b) {
                int n = Math.min(a.size(), b.size());
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    double d = a.get(i) - b.get(i);
                    sum += d * d;
                }
                return sum / (double) n;
            }

            @Override
            public double maxDistance() {
                return 255.0 * 255.0;
            }

            @Override
            public String name() {
                return "visionMSEList([0,255])";
            }
        };

        // Keep codelet responsive; adjust to your architecture tick
        setTimeStep(20);
    }

    @Override
    public void proc() {

        int episode = visionSensor.getEpoch();

        // Start a new episode when episode changes
        if (currentEpisode < 0) {
            startEpisode(episode);
        } else if (episode != currentEpisode) {
            // Finalize previous episode even if we didn't finish schedule (optional)
            finishEpisode(currentEpisode, true);
            startEpisode(episode);
        }

        switch (phase) {
            case WAIT_EPISODE_START:
                // Should not stay here; startEpisode() sets WARMUP
                break;

            case WARMUP:
                // Ensure streaming is producing fresh frames to avoid MSE=0 artifacts
                List<Float> fresh = snapshotFreshFromSensor(freshFrameTimeoutMs);
                if (fresh != null) warmupFreshFrames++;
                if (warmupFreshFrames >= 2) {
                    warmupFreshFrames = 0;
                    phase = Phase.CAPTURE_T0;
                }
                break;

            case CAPTURE_T0:
                hooks.beforeStimulus(currentEpisode, delaysMs[delayIdx], trialIdx);

                groundTruthT0 = snapshotFreshFromSensor(freshFrameTimeoutMs);
                if (groundTruthT0 == null) {
                    // Try again next tick
                    break;
                }

                cue = cueGen.sample(rnd, groundTruthT0);

                // Novo: injeta ruído no buffer logo após a captura do estímulo
                if (enableBufferNoise && APPLY_NOISE_AT_CAPTURE) {
                    injectNoiseIntoVisionBuffer();
                }

                hooks.afterStimulus(currentEpisode, delaysMs[delayIdx], trialIdx);

                retentionEndMs = System.currentTimeMillis() + delaysMs[delayIdx];
                phase = Phase.RETENTION;
                break;

            case RETENTION:
                // Novo: mantém degradação do buffer durante a retenção
                if (enableBufferNoise && APPLY_NOISE_DURING_RETENTION) {
                    for (int i = 0; i < NOISE_WRITES_PER_TICK; i++) {
                        injectNoiseIntoVisionBuffer();
                    }
                }

                if (activeInterference) {
                    induceInterferenceTick();
                    hooks.duringRetentionTick(currentEpisode, delaysMs[delayIdx], trialIdx);
                }
                if (System.currentTimeMillis() >= retentionEndMs) {
                    phase = Phase.QUERY_AND_LOG;
                }
                break;

            case QUERY_AND_LOG:
                List<Float> stored = snapshotFromBufferMO();
                if (stored == null) {
                    // Buffer not ready yet
                    break;
                }

                // Apply the SAME cue to both representations (partial report)
                List<Float> truthPatch = cue.extract(groundTruthT0);
                List<Float> storedPatch = cue.extract(stored);

                double d = metric.distance(storedPatch, truthPatch);
                double fidelity = 1.0 - (d / metric.maxDistance());
                if (Double.isFinite(fidelity)) {
                    fidelity = Math.max(0.0, Math.min(1.0, fidelity));
                } else {
                    fidelity = 0.0;
                }

                writePerTrial(currentEpisode, delaysMs[delayIdx], trialIdx, cue.describe(), d, fidelity);

                sumF[delayIdx] += fidelity;
                sumF2[delayIdx] += fidelity * fidelity;
                count[delayIdx]++;

                hooks.afterTrial(currentEpisode, delaysMs[delayIdx], trialIdx);

                advanceScheduleOrFinish();
                break;

            case FINISH_EPISODE:
                // No-op; waiting for episode increment
                break;
        }
    }

    private void startEpisode(int episode) {
        currentEpisode = episode;
        delayIdx = 0;
        trialIdx = 0;
        Arrays.fill(sumF, 0.0);
        Arrays.fill(sumF2, 0.0);
        Arrays.fill(count, 0);
        warmupFreshFrames = 0;

        openPerTrial(episode);
        hooks.onEpisodeStart(episode);

        phase = Phase.WARMUP;

        System.out.println("[VisionSperling] episode started: episode=" + episode +
                " delays=" + Arrays.toString(delaysMs) +
                " trialsPerDelay=" + trialsPerDelay +
                " interference=" + activeInterference +
                " bufferNoise=" + enableBufferNoise);
    }

    private void advanceScheduleOrFinish() {
        trialIdx++;
        if (trialIdx >= trialsPerDelay) {
            trialIdx = 0;
            delayIdx++;
        }

        if (delayIdx >= delaysMs.length) {
            finishEpisode(currentEpisode, false);
            phase = Phase.FINISH_EPISODE;
        } else {
            phase = Phase.CAPTURE_T0;
        }
    }

    private void finishEpisode(int episode, boolean aborted) {
        try {
            closePerTrial();

            // Build mean curve
            double[] meanF = new double[delaysMs.length];
            double[] stdF = new double[delaysMs.length];

            for (int k = 0; k < delaysMs.length; k++) {
                if (count[k] == 0) {
                    meanF[k] = Double.NaN;
                    stdF[k] = Double.NaN;
                } else {
                    meanF[k] = sumF[k] / count[k];
                    double var = (sumF2[k] / count[k]) - (meanF[k] * meanF[k]);
                    stdF[k] = Math.sqrt(Math.max(0.0, var));
                }
            }

            // Fit exponential decay using your lib
            DecayModelFitter.Params params = DecayModelFitter.fitExponential(delaysMs, meanF);

            // Write summary CSV
            writeSummary(episode, aborted, meanF, stdF, params);

            // Plot PNG: points + fitted curve
            writePlot(episode, aborted, meanF, params);

            hooks.onEpisodeEnd(episode);

            System.out.println("[VisionSperling] episode finished: episode=" + episode +
                    " aborted=" + aborted +
                    " F0=" + params.F0 +
                    " lambda=" + params.lambda +
                    " r2=" + params.r2 +
                    " used=" + params.usedPoints +
                    " bufferNoise=" + enableBufferNoise);

        } catch (Exception e) {
            System.err.println("[VisionSperling] finishEpisode error: " + e.getMessage());
        }
    }

    private void openPerTrial(int episode) {
        try {
            String cond = activeInterference ? "active" : "passive";
            File f = new File(outDir, "vision_sperling_per_trial_episode_" + episode + "_" + cond + ".csv");
            perTrialWriter = new PrintWriter(new FileWriter(f, false));
            perTrialWriter.println("episode,condition,delay_ms,trial_idx,cue_desc,distance_mse,fidelity");
            perTrialWriter.flush();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void closePerTrial() {
        if (perTrialWriter != null) {
            perTrialWriter.flush();
            perTrialWriter.close();
            perTrialWriter = null;
        }
    }

    private void writePerTrial(int episode, long delayMs, int trialIdx, String cueDesc, double distance, double fidelity) {
        if (perTrialWriter == null) return;
        String cond = activeInterference ? "active" : "passive";
        perTrialWriter.printf(Locale.US, "%d,%s,%d,%d,\"%s\",%.10f,%.10f%n",
                episode, cond, delayMs, trialIdx, cueDesc, distance, fidelity);
        perTrialWriter.flush();
    }

    private void writeSummary(int episode, boolean aborted, double[] meanF, double[] stdF, DecayModelFitter.Params p) throws Exception {
        String cond = activeInterference ? "active" : "passive";
        File f = new File(outDir, "vision_sperling_summary_episode_" + episode + "_" + cond + ".csv");

        try (PrintWriter pw = new PrintWriter(new FileWriter(f, false))) {
            pw.println("episode,condition,aborted,delay_ms,mean_fidelity,std_fidelity,F0,lambda,r2,used_points");
            for (int k = 0; k < delaysMs.length; k++) {
                pw.printf(Locale.US, "%d,%s,%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%d%n",
                        episode, cond, Boolean.toString(aborted), delaysMs[k],
                        meanF[k], stdF[k], p.F0, p.lambda, p.r2, p.usedPoints);
            }
        }
    }

    private void writePlot(int episode, boolean aborted, double[] meanF, DecayModelFitter.Params p) throws Exception {
        String cond = activeInterference ? "active" : "passive";
        File out = new File(outDir, "vision_sperling_plot_episode_" + episode + "_" + cond + ".png");

        int W = 900, H = 600;
        int marginL = 90, marginR = 30, marginT = 40, marginB = 90;

        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, W, H);

        int x0 = marginL;
        int y0 = H - marginB;
        int x1 = W - marginR;
        int y1 = marginT;

        g.setColor(Color.BLACK);
        g.drawLine(x0, y0, x1, y0);
        g.drawLine(x0, y0, x0, y1);

        g.drawString("Delay (ms)", (W / 2) - 30, H - 40);
        g.drawString("Mean fidelity", 20, (H / 2));

        long maxDelay = delaysMs[delaysMs.length - 1];
        if (maxDelay <= 0) maxDelay = 1;

        // points
        g.setColor(Color.BLACK);
        for (int i = 0; i < delaysMs.length; i++) {
            double f = meanF[i];
            if (!Double.isFinite(f)) continue;

            int px = x0 + (int) ((x1 - x0) * (delaysMs[i] / (double) maxDelay));
            int py = y0 - (int) ((y0 - y1) * f);
            g.fillOval(px - 4, py - 4, 8, 8);
        }

        // fitted curve
        if (Double.isFinite(p.F0) && Double.isFinite(p.lambda)) {
            g.setColor(Color.BLUE);
            int prevX = -1, prevY = -1;
            int samples = 200;
            for (int i = 0; i <= samples; i++) {
                long t = (long) (maxDelay * (i / (double) samples));
                double f = p.F0 * Math.exp(-p.lambda * (double) t);
                f = Math.max(0.0, Math.min(1.0, f));

                int px = x0 + (int) ((x1 - x0) * (t / (double) maxDelay));
                int py = y0 - (int) ((y0 - y1) * f);

                if (prevX >= 0) g.drawLine(prevX, prevY, px, py);
                prevX = px;
                prevY = py;
            }
        }

        g.setColor(Color.DARK_GRAY);
        g.drawString("Condition: " + cond + "  aborted=" + aborted, 100, 30);
        g.drawString(String.format(Locale.US, "F0=%.4f  lambda=%.6f  R2=%.4f  used=%d",
                p.F0, p.lambda, p.r2, p.usedPoints), 320, 30);

        g.dispose();
        ImageIO.write(img, "png", out);
    }

    /**
     * Read the agent's stored buffer (the object under test).
     * This must be the MemoryObject written by the vision/perception codelets.
     */
    @SuppressWarnings("unchecked")
    private List<Float> snapshotFromBufferMO() {
        Object o = visionBufferMO.getI();
        if (!(o instanceof List)) return null;
        List<Float> data = (List<Float>) o;
        if (data.size() < res * res * 3) return null;
        return new ArrayList<>(data); // defensive copy
    }

    /**
     * Read a fresh frame from VisionVrep to avoid comparing identical stale frames.
     * Returns null if no new frame is detected before timeout.
     */
    private List<Float> snapshotFreshFromSensor(long timeoutMs) {
        long start = System.currentTimeMillis();

        List<Float> last = snapshotFromSensor();
        if (last == null) return null;
        long lastSig = signature(last);

        while (System.currentTimeMillis() - start < timeoutMs) {
            try { Thread.sleep(2); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

            List<Float> cur = snapshotFromSensor();
            if (cur == null) continue;

            long curSig = signature(cur);
            if (curSig != lastSig) return cur;
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private List<Float> snapshotFromSensor() {
        Object o = visionSensor.getData();
        if (!(o instanceof List)) return null;
        return new ArrayList<>((List<Float>) o);
    }

    /**
     * Lightweight signature to detect frame changes without scanning the entire image.
     */
    private long signature(List<Float> img) {
        int n = img.size();
        int step = Math.max(1, n / 64);
        long h = 1469598103934665603L;
        for (int i = 0; i < n; i += step) {
            int v = Float.floatToIntBits(img.get(i));
            h ^= v;
            h *= 1099511628211L;
        }
        return h;
    }

    /**
     * Active interference: extra sensor reads + tiny CPU work during retention.
     * This creates competition for time/bandwidth similar to extra concurrent codelets.
     */
    private void induceInterferenceTick() {
        for (int i = 0; i < interferenceReadsPerTick; i++) {
            visionSensor.getData();
        }

        // tiny CPU load to ensure interference even when sensor returns novalue
        if (groundTruthT0 != null && groundTruthT0.size() >= res * res * 3) {
            int idx = ((res / 2) * res + (res / 2)) * 3;
            float a = groundTruthT0.get(idx);
            float b = groundTruthT0.get(idx + 1);
            float c = groundTruthT0.get(idx + 2);
            float dummy = a * 0.13f + b * 0.17f + c * 0.19f;
            if (dummy == -9999f) System.out.print("");
        }
    }

    /**
     * Novo: corrompe o conteúdo atual do visionBufferMO com ruído gaussiano.
     */
    @SuppressWarnings("unchecked")
    private void injectNoiseIntoVisionBuffer() {
        if (visionBufferMO == null) return;

        Object o = visionBufferMO.getI();
        if (!(o instanceof List)) return;

        List<Float> original = (List<Float>) o;
        if (original.size() < res * res * 3) return;

        ArrayList<Float> noisy = new ArrayList<>(original.size());

        for (int i = 0; i < original.size(); i++) {
            float v = original.get(i);
            v = applyGaussianNoise(v);
            noisy.add(clamp255(v));
        }

        visionBufferMO.setI(noisy);
    }

    private float applyGaussianNoise(float v) {
        return (float) (v + rnd.nextGaussian() * BUFFER_GAUSSIAN_STD);
    }

    private float clamp255(float v) {
        if (v < 0f) return 0f;
        if (v > 255f) return 255f;
        return v;
    }

    @Override
    public void stop() {
        try { closePerTrial(); } catch (Exception ignored) {}
        super.stop();
    }
}