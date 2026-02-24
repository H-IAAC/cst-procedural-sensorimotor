package codelets.sensors;

import CommunicationInterface.SensorI;
import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import javax.imageio.ImageIO;

/**
 * Specialized, publishable Sperling-inspired benchmark for VisionVrep.
 *
 * Key properties:
 * - Runs incrementally inside CST (state machine in proc()).
 * - Integrates with real episodes by tracking vision.getEpoch() changes.
 * - Measures buffer fidelity (ground-truth from VisionVrep sensor vs stored buffer in MemoryObject).
 * - Supports active interference condition.
 * - Exports per-trial CSV, summary CSV, and an automatic PNG plot per episode.
 */
public class VisionVrepSperling extends Codelet {

    @Override
    public void accessMemoryObjects() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public void calculateActivation() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    private enum Phase {
        WAIT_NEW_EPISODE,
        WARMUP_FRAMES,
        CAPTURE_T0,
        RETENTION,
        QUERY_AND_LOG,
        FINISHED_EPISODE
    }

    private final SensorI visionSensor;          // external reference (proxy ground truth)
    private final MemoryObject visionBufferMO;      // internal agent buffer (List<Float>)
    private final int res;
    private final int patchSize;
    private final long[] delaysMs;
    private final int trialsPerDelay;
    private final boolean activeInterference;
    private final int interferenceReadsPerTick;
    private final long frameTimeoutMs;
    private final File outDir;
    private final Random rnd;

    // Episode tracking
    private int currentEpoch = -1;
    private boolean episodeDoneForThisEpoch = false;

    // Trial scheduling
    private Phase phase = Phase.WAIT_NEW_EPISODE;
    private int warmupOkFrames = 0;

    private int delayIdx = 0;
    private int trialIdx = 0;

    private long retentionEndMs = 0;

    // Trial data
    private List<Float> groundTruthFull = null;
    private int x0 = 0, y0 = 0;

    // Online summaries per delay
    private final double[] sumF;
    private final double[] sumF2;
    private final int[] count;

    // Writers (per episode)
    private PrintWriter perTrialWriter = null;

    public VisionVrepSperling(
            SensorI visionSensor,
            MemoryObject visionBufferMO,
            int res,
            int patchSize,
            long[] delaysMs,
            int trialsPerDelay,
            boolean activeInterference,
            int interferenceReadsPerTick,
            long frameTimeoutMs,
            File outDir,
            long seed
    ) {
        this.visionSensor = Objects.requireNonNull(visionSensor, "visionSensor");
        this.visionBufferMO = visionBufferMO; // may be null (falls back to sensor), but for publishable benchmarks pass it
        this.res = res;
        this.patchSize = patchSize;
        this.delaysMs = Arrays.copyOf(delaysMs, delaysMs.length);
        this.trialsPerDelay = trialsPerDelay;
        this.activeInterference = activeInterference;
        this.interferenceReadsPerTick = Math.max(0, interferenceReadsPerTick);
        this.frameTimeoutMs = Math.max(50, frameTimeoutMs);
        this.outDir = outDir == null ? new File("vision_sperling_out") : outDir;
        this.rnd = new Random(seed);

        this.sumF = new double[delaysMs.length];
        this.sumF2 = new double[delaysMs.length];
        this.count = new int[delaysMs.length];

        if (!this.outDir.exists()) this.outDir.mkdirs();

        // Make the codelet reasonably frequent; adjust if needed
        setTimeStep(20);
    }

    @Override
    public void proc() {

        int epoch = visionSensor.getEpoch();

        // If epoch changed, reset state for the new episode.
        if (currentEpoch < 0) {
            currentEpoch = epoch;
            episodeDoneForThisEpoch = false;
            phase = Phase.WARMUP_FRAMES;
            openEpisodeWriters(epoch);
            resetEpisodeAccumulators();
        } else if (epoch != currentEpoch) {
            // New episode started in the environment
            finalizeEpisode(currentEpoch);
            currentEpoch = epoch;
            episodeDoneForThisEpoch = false;
            phase = Phase.WARMUP_FRAMES;
            openEpisodeWriters(epoch);
            resetEpisodeAccumulators();
        }

        // If we already finished this epoch, just wait until the next epoch change.
        if (episodeDoneForThisEpoch) {
            phase = Phase.WAIT_NEW_EPISODE;
            return;
        }

        switch (phase) {

            case WAIT_NEW_EPISODE:
                // No-op; epoch change will move us to WARMUP_FRAMES
                break;

            case WARMUP_FRAMES:
                // Ensure streaming is alive: wait for a few frame changes before running trials.
                if (waitForFreshFrame(frameTimeoutMs) != null) {
                    warmupOkFrames++;
                }
                if (warmupOkFrames >= 2) {
                    warmupOkFrames = 0;
                    phase = Phase.CAPTURE_T0;
                }
                break;

            case CAPTURE_T0:
                groundTruthFull = waitForFreshFrame(frameTimeoutMs);
                if (groundTruthFull == null) {
                    // No fresh frame; try again next proc()
                    break;
                }

                // Choose a random patch once per trial; same patch applied to buffer and truth
                x0 = rnd.nextInt(res - patchSize + 1);
                y0 = rnd.nextInt(res - patchSize + 1);

                retentionEndMs = System.currentTimeMillis() + delaysMs[delayIdx];
                phase = Phase.RETENTION;
                break;

            case RETENTION:
                // Non-blocking retention interval
                if (activeInterference) {
                    induceInterferenceTick();
                }
                if (System.currentTimeMillis() >= retentionEndMs) {
                    phase = Phase.QUERY_AND_LOG;
                }
                break;

            case QUERY_AND_LOG:
                List<Float> storedFull = snapshotFromBuffer();
                if (storedFull == null || storedFull.size() < res * res * 3) {
                    // Buffer not ready; retry next proc()
                    break;
                }

                List<Float> truthPatch = extractPatch(groundTruthFull, x0, y0);
                List<Float> storedPatch = extractPatch(storedFull, x0, y0);

                double mse = computeMSE(storedPatch, truthPatch);
                double fidelity = 1.0 - (mse / (255.0 * 255.0));
                if (Double.isFinite(fidelity)) {
                    fidelity = Math.max(0.0, Math.min(1.0, fidelity));
                } else {
                    fidelity = 0.0;
                }

                logTrial(currentEpoch, delayIdx, trialIdx, x0, y0, mse, fidelity);

                // Update online statistics
                sumF[delayIdx] += fidelity;
                sumF2[delayIdx] += fidelity * fidelity;
                count[delayIdx]++;

                advanceSchedule();
                break;

            case FINISHED_EPISODE:
                // Should not stay here; finalizeEpisode() will switch to WAIT_NEW_EPISODE
                break;
        }
    }

    private void advanceSchedule() {
        trialIdx++;

        if (trialIdx >= trialsPerDelay) {
            trialIdx = 0;
            delayIdx++;
        }

        if (delayIdx >= delaysMs.length) {
            // Episode schedule completed
            finalizeEpisode(currentEpoch);
            episodeDoneForThisEpoch = true;
            phase = Phase.WAIT_NEW_EPISODE;
        } else {
            phase = Phase.CAPTURE_T0;
        }
    }

    private void resetEpisodeAccumulators() {
        Arrays.fill(sumF, 0.0);
        Arrays.fill(sumF2, 0.0);
        Arrays.fill(count, 0);
        delayIdx = 0;
        trialIdx = 0;
    }

    private void openEpisodeWriters(int epoch) {
        closeEpisodeWriters();

        try {
            String condition = activeInterference ? "active" : "passive";
            File perTrial = new File(outDir, "vision_sperling_per_trial_epoch_" + epoch + "_" + condition + ".csv");
            perTrialWriter = new PrintWriter(new FileWriter(perTrial, false));
            perTrialWriter.println("epoch,condition,delay_ms,trial_idx,x0,y0,mse,fidelity");
            perTrialWriter.flush();
        } catch (Exception e) {
            throw new RuntimeException("Failed to open CSV writers", e);
        }
    }

    private void closeEpisodeWriters() {
        if (perTrialWriter != null) {
            perTrialWriter.flush();
            perTrialWriter.close();
            perTrialWriter = null;
        }
    }

    private void logTrial(int epoch, int delayIndex, int trialIndex, int x0, int y0, double mse, double fidelity) {
        if (perTrialWriter == null) return;
        String condition = activeInterference ? "active" : "passive";
        perTrialWriter.printf(Locale.US,
                "%d,%s,%d,%d,%d,%d,%.10f,%.10f%n",
                epoch, condition, delaysMs[delayIndex], trialIndex, x0, y0, mse, fidelity
        );
        perTrialWriter.flush();
    }

    private void finalizeEpisode(int epoch) {
        // Compute mean fidelity per delay
        double[] meanF = new double[delaysMs.length];
        double[] stdF = new double[delaysMs.length];

        for (int k = 0; k < delaysMs.length; k++) {
            if (count[k] == 0) {
                meanF[k] = Double.NaN;
                stdF[k] = Double.NaN;
                continue;
            }
            meanF[k] = sumF[k] / count[k];
            double var = (sumF2[k] / count[k]) - (meanF[k] * meanF[k]);
            stdF[k] = Math.sqrt(Math.max(0.0, var));
        }

        // Fit exponential decay on mean fidelities
        DecayFit fit = fitExponential(delaysMs, meanF);

        // Write summary CSV
        try {
            String condition = activeInterference ? "active" : "passive";
            File summary = new File(outDir, "vision_sperling_summary_epoch_" + epoch + "_" + condition + ".csv");
            try (PrintWriter pw = new PrintWriter(new FileWriter(summary, false))) {
                pw.println("epoch,condition,delay_ms,mean_fidelity,std_fidelity,F0,lambda,r2,used_points");
                for (int k = 0; k < delaysMs.length; k++) {
                    pw.printf(Locale.US,
                            "%d,%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%d%n",
                            epoch, condition, delaysMs[k],
                            meanF[k], stdF[k],
                            fit.F0, fit.lambda, fit.r2, fit.usedPoints
                    );
                }
            }
        } catch (Exception e) {
            System.err.println("[VisionVrepSperlingCodelet] Failed to write summary CSV: " + e.getMessage());
        }

        // Generate plot (mean points + fitted curve)
        try {
            String condition = activeInterference ? "active" : "passive";
            File plot = new File(outDir, "vision_sperling_plot_epoch_" + epoch + "_" + condition + ".png");
            generatePlot(plot, delaysMs, meanF, fit);
        } catch (Exception e) {
            System.err.println("[VisionVrepSperlingCodelet] Failed to generate plot: " + e.getMessage());
        }

        closeEpisodeWriters();
        phase = Phase.FINISHED_EPISODE;
    }

    /**
     * Returns a fresh frame from the sensor (VisionVrep), or null if no update arrives within timeout.
     */
    private List<Float> waitForFreshFrame(long timeoutMs) {
        long start = System.currentTimeMillis();

        List<Float> last = snapshotFromSensor();
        if (last == null) return null;
        long lastSig = signature(last);

        while (System.currentTimeMillis() - start < timeoutMs) {
            // We avoid long sleep; CST scheduling will call proc() frequently anyway.
            // Still, a tiny pause helps reduce CPU usage if getData() is cheap here.
            try { Thread.sleep(2); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

            List<Float> cur = snapshotFromSensor();
            if (cur == null) continue;
            long curSig = signature(cur);

            if (curSig != lastSig) {
                return cur;
            }
        }

        return null;
    }

    @SuppressWarnings("unchecked")
    private List<Float> snapshotFromSensor() {
        Object o = visionSensor.getData();
        if (!(o instanceof List)) return null;
        return new ArrayList<>((List<Float>) o);
    }

    @SuppressWarnings("unchecked")
    private List<Float> snapshotFromBuffer() {
        if (visionBufferMO == null) {
            // Fallback: if you did not provide a buffer MemoryObject, we use the sensor again.
            // This is NOT buffer fidelity; it becomes a sensor stream stability test.
            return snapshotFromSensor();
        }
        Object o = visionBufferMO.getI();
        if (!(o instanceof List)) return null;
        return new ArrayList<>((List<Float>) o);
    }

    /**
     * Lightweight signature to detect frame changes.
     */
    private long signature(List<Float> img) {
        if (img == null || img.isEmpty()) return 0L;
        int n = img.size();
        int step = Math.max(1, n / 64); // sample ~64 values
        long h = 1469598103934665603L;  // FNV offset basis
        for (int i = 0; i < n; i += step) {
            int v = Float.floatToIntBits(img.get(i));
            h ^= v;
            h *= 1099511628211L;        // FNV prime
        }
        return h;
    }

    private void induceInterferenceTick() {
        // Interference strategy: extra sensor reads (remote calls + memory copy cost).
        // This stresses timing/bandwidth and competes with other codelets.
        for (int i = 0; i < interferenceReadsPerTick; i++) {
            visionSensor.getData();
        }

        // Optional cheap CPU load (very small): compute a tiny checksum on a patch.
        // Keeps interference "active" even if the sensor returns novalue.
        if (groundTruthFull != null && groundTruthFull.size() >= res * res * 3) {
            int idx = ((y0 * res) + x0) * 3;
            float a = groundTruthFull.get(idx);
            float b = groundTruthFull.get(idx + 1);
            float c = groundTruthFull.get(idx + 2);
            float dummy = (a * 0.13f + b * 0.17f + c * 0.19f);
            if (dummy == -9999f) System.out.print(""); // never true; prevents JIT from removing completely
        }
    }

    private List<Float> extractPatch(List<Float> img, int x0, int y0) {
        ArrayList<Float> patch = new ArrayList<>(patchSize * patchSize * 3);

        for (int y = y0; y < y0 + patchSize; y++) {
            for (int x = x0; x < x0 + patchSize; x++) {
                int idx = (y * res + x) * 3;
                patch.add(img.get(idx));
                patch.add(img.get(idx + 1));
                patch.add(img.get(idx + 2));
            }
        }
        return patch;
    }

    private double computeMSE(List<Float> a, List<Float> b) {
        double sum = 0.0;
        int n = Math.min(a.size(), b.size());
        for (int i = 0; i < n; i++) {
            double d = a.get(i) - b.get(i);
            sum += d * d;
        }
        return sum / (double) n;
    }

    private static final class DecayFit {
        final double F0;
        final double lambda;
        final double r2;
        final int usedPoints;

        DecayFit(double F0, double lambda, double r2, int usedPoints) {
            this.F0 = F0;
            this.lambda = lambda;
            this.r2 = r2;
            this.usedPoints = usedPoints;
        }

        double predict(long tMs) {
            return F0 * Math.exp(-lambda * (double) tMs);
        }
    }

    /**
     * Fits ln(F) = a + b*t, where lambda = -b and F0 = exp(a).
     * Computes R^2 in log-space.
     */
    private DecayFit fitExponential(long[] t, double[] F) {
        ArrayList<Double> xs = new ArrayList<>();
        ArrayList<Double> ys = new ArrayList<>();

        for (int i = 0; i < t.length; i++) {
            double f = F[i];
            if (Double.isFinite(f) && f > 0) {
                xs.add((double) t[i]);
                ys.add(Math.log(f));
            }
        }

        int n = xs.size();
        if (n < 2) return new DecayFit(Double.NaN, Double.NaN, Double.NaN, n);

        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++) { xMean += xs.get(i); yMean += ys.get(i); }
        xMean /= n; yMean /= n;

        double sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) {
            double dx = xs.get(i) - xMean;
            double dy = ys.get(i) - yMean;
            sxx += dx * dx;
            sxy += dx * dy;
        }
        if (sxx == 0) return new DecayFit(Double.NaN, Double.NaN, Double.NaN, n);

        double b = sxy / sxx;
        double a = yMean - b * xMean;

        double F0 = Math.exp(a);
        double lambda = -b;

        // R^2 in log-space
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            double yi = ys.get(i);
            double yHat = a + b * xs.get(i);
            ssTot += (yi - yMean) * (yi - yMean);
            ssRes += (yi - yHat) * (yi - yHat);
        }
        double r2 = (ssTot == 0) ? Double.NaN : (1.0 - ssRes / ssTot);

        return new DecayFit(F0, lambda, r2, n);
    }

    private void generatePlot(File outFile, long[] delays, double[] meanF, DecayFit fit) throws Exception {
        int W = 900, H = 600;
        int marginL = 90, marginR = 30, marginT = 40, marginB = 90;

        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // background
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, W, H);

        // axes
        g.setColor(Color.BLACK);
        int x0 = marginL;
        int y0 = H - marginB;
        int x1 = W - marginR;
        int y1 = marginT;

        g.drawLine(x0, y0, x1, y0);
        g.drawLine(x0, y0, x0, y1);

        // labels
        g.drawString("Delay (ms)", (W / 2) - 30, H - 40);
        g.drawString("Mean fidelity", 20, (H / 2));

        long maxDelay = delays[delays.length - 1];
        if (maxDelay <= 0) maxDelay = 1;

        // points
        g.setColor(Color.BLACK);
        for (int i = 0; i < delays.length; i++) {
            double f = meanF[i];
            if (!Double.isFinite(f)) continue;

            int px = x0 + (int) ((x1 - x0) * (delays[i] / (double) maxDelay));
            int py = y0 - (int) ((y0 - y1) * f);
            g.fillOval(px - 4, py - 4, 8, 8);
        }

        // fitted curve
        if (Double.isFinite(fit.F0) && Double.isFinite(fit.lambda)) {
            g.setColor(Color.BLUE);
            int prevX = -1, prevY = -1;
            int samples = 200;
            for (int i = 0; i <= samples; i++) {
                long t = (long) (maxDelay * (i / (double) samples));
                double f = fit.predict(t);
                f = Math.max(0.0, Math.min(1.0, f));

                int px = x0 + (int) ((x1 - x0) * (t / (double) maxDelay));
                int py = y0 - (int) ((y0 - y1) * f);

                if (prevX >= 0) g.drawLine(prevX, prevY, px, py);
                prevX = px; prevY = py;
            }
        }

        // annotation
        g.setColor(Color.DARK_GRAY);
        String cond = activeInterference ? "active" : "passive";
        g.drawString("Condition: " + cond, 100, 30);
        g.drawString(String.format(Locale.US, "F0=%.4f  lambda=%.6f  R2=%.4f  used=%d",
                fit.F0, fit.lambda, fit.r2, fit.usedPoints), 280, 30);

        g.dispose();
        ImageIO.write(img, "png", outFile);
    }

    @Override
    public void stop() {
        closeEpisodeWriters();
        super.stop();
    }
}
