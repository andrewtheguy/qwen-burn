/// Mel spectrogram computation for Qwen3-ASR.
/// FFT adapted from candle whisper/audio.rs. Slaney mel filterbank for 128 bins.
use std::sync::Arc;
use std::thread;

const SAMPLE_RATE: usize = 16000;
const NUM_MEL_BINS: usize = 128;
const HOP_LENGTH: usize = 160;
const N_FFT: usize = 400;
const MEL_FMAX: f32 = 8000.0;

// ── FFT (from whisper/audio.rs) ─────────────────────────────────────────────

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![0.0f32; n * 2];
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);
    for (i, &v) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(v);
        } else {
            odd.push(v);
        }
    }
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);
    let two_pi = 2.0 * std::f32::consts::PI;
    let n_f = n as f32;
    for k in 0..n / 2 {
        let theta = two_pi * k as f32 / n_f;
        let re = theta.cos();
        let im = -theta.sin();
        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn dft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    let n_f = n as f32;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &v) in inp.iter().enumerate() {
            let angle = two_pi * k as f32 * j as f32 / n_f;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

// ── Slaney mel filterbank ───────────────────────────────────────────────────

fn hertz_to_mel(freq: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 27.0 / 1.856_298; // 27 / ln(6.4)
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
    } else {
        3.0 * freq / 200.0
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 1.856_298 / 27.0; // ln(6.4) / 27
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// Compute Slaney-style mel filterbank [n_fft/2+1, NUM_MEL_BINS] stored row-major.
/// filters[freq * NUM_MEL_BINS + mel] = weight
pub fn compute_mel_filters() -> Vec<f32> {
    let num_freq_bins = 1 + N_FFT / 2; // 201
    let fft_freqs: Vec<f32> = (0..num_freq_bins)
        .map(|i| i as f32 * SAMPLE_RATE as f32 / 2.0 / (num_freq_bins - 1) as f32)
        .collect();

    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(MEL_FMAX);
    let mel_freqs: Vec<f32> = (0..NUM_MEL_BINS + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (NUM_MEL_BINS + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect();

    let filter_diff: Vec<f32> = mel_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    // fb[freq, mel] = max(0, min(down_slope, up_slope)) * enorm
    // down_slope = rising edge: (fft_freq - left) / (center - left)
    // up_slope = falling edge: (right - fft_freq) / (right - center)
    let mut fb = vec![0.0f32; num_freq_bins * NUM_MEL_BINS];
    for f in 0..num_freq_bins {
        for m in 0..NUM_MEL_BINS {
            let down_slope = (fft_freqs[f] - mel_freqs[m]) / filter_diff[m];
            let up_slope = (mel_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            let val = down_slope.min(up_slope).max(0.0);
            let enorm = 2.0 / (mel_freqs[m + 2] - mel_freqs[m]);
            fb[f * NUM_MEL_BINS + m] = val * enorm;
        }
    }
    fb
}

// ── STFT + mel spectrogram ──────────────────────────────────────────────────

fn log_mel_spectrogram_worker(
    ith: usize,
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    n_len: usize,
    n_threads: usize,
) -> Vec<f32> {
    let n_fft_half = 1 + N_FFT / 2; // 201
    let mut fft_in = vec![0.0f32; N_FFT];
    let mut mel = vec![0.0f32; n_len * NUM_MEL_BINS];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / HOP_LENGTH + 1, n_len);

    for i in (ith..end).step_by(n_threads) {
        let offset = i * HOP_LENGTH;

        // Apply Hann window
        let copy_len = std::cmp::min(N_FFT, n_samples.saturating_sub(offset));
        for j in 0..copy_len {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        for v in fft_in.iter_mut().take(N_FFT).skip(copy_len) {
            *v = 0.0;
        }

        // FFT
        let mut fft_out = fft(&fft_in);

        // |FFT|^2 for positive frequencies only (0..n_fft/2+1)
        // No fold — matches torch.stft(..., return_complex=True).abs()**2
        for j in 0..n_fft_half {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }

        // Mel filterbank: filters is [n_fft_half, NUM_MEL_BINS]
        for m in 0..NUM_MEL_BINS {
            let mut sum = 0.0f32;
            for f in 0..n_fft_half {
                sum += fft_out[f] * filters[f * NUM_MEL_BINS + m];
            }
            mel[m * n_len + i] = sum.max(1e-10).log10();
        }
    }
    mel
}

/// Compute mel spectrogram from raw PCM f32 samples (16kHz mono).
/// Returns flat [NUM_MEL_BINS, n_frames] in row-major order.
pub fn compute_mel_spectrogram(samples: &[f32]) -> Vec<f32> {
    let filters = compute_mel_filters();
    let n_len = samples.len() / HOP_LENGTH;
    // No padding — just compute exact frames

    // Each thread computes a strided subset of STFT frames (thread i handles frames i, i+n, i+2n, ...).
    // Results are summed across threads since each thread writes to non-overlapping frame indices.
    let n_threads = std::thread::available_parallelism().map_or(2, |n| n.get());

    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    let hann = Arc::new(hann);
    let samples_arc = Arc::new(samples.to_vec());
    let filters = Arc::new(filters);

    let all_outputs = thread::scope(|s| {
        (0..n_threads)
            .map(|tid| {
                let hann = Arc::clone(&hann);
                let samples = Arc::clone(&samples_arc);
                let filters = Arc::clone(&filters);
                s.spawn(move || {
                    log_mel_spectrogram_worker(tid, &hann, &samples, &filters, n_len, n_threads)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("mel thread panicked"))
            .collect::<Vec<_>>()
    });

    // Sum thread outputs
    let l = all_outputs[0].len();
    let mut mel = vec![0.0f32; l];
    for out in &all_outputs {
        for (i, &v) in out.iter().enumerate() {
            mel[i] += v;
        }
    }

    // Dynamic clamping and normalization: max(mel, global_max - 8) then (x + 4) / 4
    let mmax = mel
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        - 8.0;
    for m in mel.iter_mut() {
        *m = m.max(mmax);
        *m = (*m + 4.0) / 4.0;
    }

    mel
}

/// Number of mel frames for a given sample count.
pub fn mel_frames(n_samples: usize) -> usize {
    n_samples / HOP_LENGTH
}
