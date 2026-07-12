
# Automatic Modulation Classification (AMC) using Deep Learning and a Software-Defined Radio Pipeline

**Research Internship Project — Indian Institute of Technology, Roorkee**
**Department of Electronics and Communication Engineering**

Author: **Mansi Ambwani** · Supervisor: **Dr. Anshul Jaiswal**

> An end-to-end system that transmits and receives eight digital modulation schemes over the air using a pair of USRP software-defined radios, and classifies them in real time using a custom 1-D residual CNN (**RadioResNet95**) trained on physically captured I/Q samples.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Modulation Schemes Covered](#modulation-schemes-covered)
6. [Model — RadioResNet95](#model--radioresnet95)
7. [Hardware & Software Requirements](#hardware--software-requirements)
8. [Setup and Usage](#setup-and-usage)
9. [Known Issue & Recommended Fix](#known-issue--recommended-fix)
10. [Challenges Faced](#challenges-faced)
11. [Future Work](#future-work)
12. [Report](#report)
13. [References](#references)

---

## Overview

Automatic Modulation Classification (AMC) enables a receiver to identify the modulation scheme of an incoming signal without prior coordination with the transmitter — a capability central to cognitive radio, spectrum monitoring, electronic warfare, and blind receiver design.

This project builds a **complete, working AMC pipeline** across three tightly coupled components:

| Component | Description |
|---|---|
| **SDR Hardware Pipeline** | A matched pair of USRP radios transmits and receives 8 modulation schemes over the air at a 1.5 GHz carrier frequency, with a full receive-side DSP chain (frame sync, CFO correction, equalization, Costas-loop phase tracking). |
| **Dataset Generation** | Received symbols are synchronized, equalized, and stored as raw I/Q samples, forming a labeled 8-class dataset. |
| **CNN Classifier — RadioResNet95** | A 1-D residual CNN trained on 1024-sample I/Q windows to classify the received signal's modulation scheme. |
| **Real-Time Integration** | A single integrated script that drives the USRP pair live, runs the full DSP + inference pipeline per frame, and reports a running confusion matrix. |

The physical transmit/receive link was independently validated and achieved a **near-zero Bit Error Rate (BER)** across nearly all eight modulations, confirming the hardware chain itself is reliable before any classification results are considered.

---

## Key Results

Three evaluations of RadioResNet95 are reported (see the full report for details):

| Evaluation | Setup | Overall Accuracy |
|---|---|---|
| **Offline held-out test** | 331 samples held out from the original captured dataset (same session as training data) | **99.70%** |
| **Real-time, over-the-air (USRP)** | 10 fresh frames/modulation, transmitted & received live, majority-vote per frame | **63.75%** |
| **Balanced-dataset offline test** | Held-out split from a much larger, class-balanced dataset (150,000 windows/class) | **89.74%** |

**Pattern observed:** the entire **PSK family (BPSK, QPSK, 8-PSK, 16-PSK)** and **32-QAM** classify reliably (90–100%) in real-time testing. The three **square, Gray-coded QAM orders (16-QAM, 64-QAM, 256-QAM)** are frequently confused with one another live on air — this is diagnosed in the report as a likely amplitude-normalization artifact in the receiver DSP chain, not a fundamental limitation of the CNN (see [Known Issue](#known-issue--recommended-fix)).

---

## System Architecture

```
 Bit Generation → Modulation Mapping → RRC Pulse Shaping → USRP Transmit
        ↓
   Wireless Channel (1.5 GHz, over the air)
        ↓
 USRP Receive → Frame Sync (correlation) → CFO Correction → Channel Equalization
        ↓
 Costas-Loop Phase Tracking → Symbol Recovery → 1024-sample I/Q Windowing
        ↓
   RadioResNet95 (CNN) → Per-Window Prediction → Majority Vote → Final Class
```

The **same transceiver logic** (frame structure, pulse shaping, DSP recovery chain) is shared between two contexts:
- **Offline scripts** (one per modulation) used to generate and label the training dataset.
- **The final real-time script**, which cycles through all eight modulations live and classifies each one using the trained model.

---

## Repository Structure

```
amc-sdr-internship/
│
├── AMC_report_MansiAmbwani.pdf        Full internship report (28 pages) — methodology,
│                                        math, architecture, and full results
│
├── README.md                          This file
│
├── hardware pipeline.zip              Per-modulation USRP TX/RX + DSP scripts (offline
│                                        dataset generation), one script per scheme:
│                                          bpsk.py, qpsk.py, 8psk.py, 16psk.py,
│                                          16qam.py, 32qam.py, 64qam.py, 256qam.py
│
├── 1st best model for 8 modulations.zip
│                                       Best-performing real-time classifier:
│                                          model12.py                 (integrated TX/RX + inference script)
│                                          suyash_resnet_weights.pth  (trained RadioResNet95 weights)
│                                          Untitled17.ipynb           (training notebook)
│                                          usrp_ml_confusion_matrix.png
│
├── 1st model with HOC.zip             Variant using Higher-Order-Cumulant (HOC) features
│                                          hoc_1st.py / hoc_1st.ipynb
│                                          hoc_1st.pth.zip
│                                          realtime_8classhoc_confusion_matrix_v8.png
│
├── 1st model with overlapping.zip     Variant using overlapping I/Q windows
│                                          1st+overlap.py / 1st+overlap.ipynb
│                                          1st.pth.zip
│                                          realtime_8class_confusion_matrix_v8.png
│
├── 1st model wth hoc+overlapping.zip  Variant combining HOC features + overlapping windows
│                                          overlap+hoc_n.py / overlap+hoc_n.ipynb
│                                          ovelap+hoc_n.pth.zip
│                                          realtime_8class_confusion_matrix_v10.png
│
├── 1.5all_data.zip                    Large, class-balanced dataset — 150,000 (1.5 lakh)
│                                        raw I/Q samples per class, used for the
│                                        supplementary offline evaluation
│
└── ml_*_samples.npy                   Raw I/Q sample arrays per modulation (unzipped copies
                                         of the files inside 1.5all_data.zip), one file per class:
                                           ml_bpsk_samples.npy    ml_16qam_samples.npy
                                           ml_qpsk_samples.npy    ml_32qam_samples.npy
                                           ml_8psk_samples.npy    ml_64qam_samples.npy
                                           ml_16psk_samples.npy   ml_256qam_samples.npy
```

> **Note:** Several `.pth` checkpoints are themselves zipped inside their model folders (e.g. `hoc_1st.pth.zip`) to keep them under GitHub's file-size limits — extract them before loading with PyTorch.

---

## Modulation Schemes Covered

| Family | Schemes |
|---|---|
| **Phase Shift Keying (PSK)** | BPSK (2-PSK), QPSK (4-PSK), 8-PSK, 16-PSK |
| **Quadrature Amplitude Modulation (QAM)** | 16-QAM, 32-QAM, 64-QAM, 256-QAM |

These span both **constant-envelope** (PSK) and **variable-envelope** (QAM) modulation families, deliberately chosen to stress-test the classifier across a spectrum of separability.

---

## Model — RadioResNet95

A 1-D residual convolutional network built from scratch for this project:

- **Front-end:** a 2-D convolutional layer with a (2×7) kernel that jointly processes the I and Q rows, producing 64 feature channels.
- **Residual tower:** four stages, two residual blocks each, channel width doubling and temporal resolution halving per stage: **64 → 128 → 256 → 512**.
- **Head:** global average pooling → fully connected (512 → 128 → 8) with batch norm, ReLU, and dropout (p = 0.3).
- **Parameters:** ~3.91M trainable parameters, with the deepest 512-channel stage accounting for the majority of the count.

**Training configuration:**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.001 (cosine annealing, T_max = 40) |
| Weight Decay | 1e-4 |
| Loss | Cross-Entropy |
| Batch Size | 32 |
| Epochs | 40 |
| Train/Test Split | 80% / 20%, stratified |
| Input | 1024-sample I/Q windows, shape (N, 1, 2, 1024) |

---

## Hardware & Software Requirements

**Hardware**
- 2× USRP (Universal Software Radio Peripheral) devices — one transmitter, one receiver
- A host PC with UHD (USRP Hardware Driver) installed
- RF cabling / antennas for over-the-air transmission

**Software**
- Python 3.x
- [`uhd`](https://github.com/EttusResearch/uhd) — USRP driver bindings
- `numpy`, `scipy` — signal processing
- `matplotlib` — plotting / confusion matrices
- `torch` (PyTorch) — model training and inference
- Jupyter (to run the included `.ipynb` training notebooks)

```bash
pip install numpy scipy matplotlib torch
# UHD/uhd must be installed separately following Ettus Research's instructions
# for your OS: https://files.ettus.com/manual/page_install.html
```

---

## Setup and Usage

1. **Clone the repository** and extract the zipped folders you need:
   ```bash
   git clone https://github.com/mansi-ambwani/amc-sdr-internship.git
   cd amc-sdr-internship
   unzip "hardware pipeline.zip"
   unzip "1st best model for 8 modulations.zip"
   ```

2. **Generate a dataset (optional — a pre-captured dataset is already included as `.npy` files):**
   Run the per-modulation scripts in `hardware pipeline/` (e.g. `python bpsk.py`) with both USRP radios connected, one modulation at a time, to transmit/receive and log I/Q samples.

3. **Train the model:**
   Open the training notebook inside the model folder of your choice (e.g. `1st best model for 8 modulations/Untitled17.ipynb`) to reproduce training on the captured dataset, or load the provided pre-trained weights (`suyash_resnet_weights.pth`) directly.

4. **Run real-time classification:**
   ```bash
   python "1st best model for 8 modulations/model12.py"
   ```
   This drives both USRP radios live: for each modulation it transmits a fresh random payload, receives and processes it through the full sync/CFO/equalization pipeline, classifies each 1024-sample window with RadioResNet95, takes a majority vote per frame, and compiles a running confusion matrix at the end of the run.

> Update the UHD driver path at the top of each script (e.g. `os.add_dll_directory(...)`) to match your local UHD installation.

---

## Known Issue & Recommended Fix

The real-time square-QAM confusion (16-QAM ↔ 64-QAM ↔ 256-QAM) was traced to the receiver's **amplitude/gain-normalization step**, which originally estimated RMS gain from only the **first 40 payload symbols**. For square Gray-coded QAM constellations, a small random sample of symbols can be biased toward the inner or outer ring, skewing the RMS estimate and rescaling the *entire* frame — inflating or shrinking the effective constellation size and causing misclassification. PSK is immune (constant envelope) and 32-QAM's cross constellation has a smaller inner/outer amplitude ratio, so both stayed within the slicer's decision boundaries.

**Recommended fixes (see `model12.py` and Section 10.2 of the report):**
- Widen the gain-estimation window to use most of the received payload (not a fixed small count), letting the RMS estimate converge to its true value.
- Use a more robust estimator (e.g. median- or trimmed-mean-based amplitude estimation) that is less sensitive to which symbols are sampled first.
- Increase training dataset size per class to reduce validation variance, particularly for higher-order QAM.

---

## Challenges Faced

- **Hardware synchronization:** reliably locating frame boundaries and correcting carrier frequency offset (CFO) over the air required careful tuning of correlation-based sync and CFO estimation under real hardware impairments.
- **Square-QAM amplitude normalization:** a subtle, systematic bias affecting only square-QAM orders (see above) was uncovered only through careful confusion-matrix analysis.
- **Limited dataset size:** the original captured dataset (~146 windows/class, 1,168 total) is modest for deep learning, causing noticeable epoch-to-epoch validation fluctuation.
- **Isolating hardware vs. model issues:** since near-zero BER confirmed the physical link was reliable, extra care was needed to determine whether square-QAM confusion originated in the DSP/normalization stage or the CNN itself.

---

## Future Work

- Run a controlled ablation with a widened gain-estimation window and re-test real-time accuracy.
- Evaluate a more robust (median/trimmed-mean) amplitude estimator.
- Re-run the full real-time hardware test using the model trained on the larger, balanced dataset (150,000 windows/class).
- Expand the training dataset further to improve generalization for higher-order QAM classes.

---

## Report

The complete methodology, mathematical derivations (RRC pulse shaping, CFO estimation, Costas loop, AdamW/cosine-annealing schedule), architecture diagrams, and full per-class results are documented in:

📄 **[`AMC_report_MansiAmbwani.pdf`](./AMC_report_MansiAmbwani.pdf)** — *Automatic Modulation Classification Using Deep Learning and a Software-Defined Radio Pipeline for Eight PSK and QAM Modulation Schemes*, IIT Roorkee, Department of Electronics and Communication Engineering.

---

## References

1. T. J. O'Shea, T. Roy, and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE Journal of Selected Topics in Signal Processing*, 2018.
2. T. J. O'Shea and N. West, "Radio Machine Learning Dataset Generation with GNU Radio," *Proceedings of the GNU Radio Conference*, 2016.
3. O. A. Dobre, A. Abdi, Y. Bar-Ness, and W. Su, "Survey of Automatic Modulation Classification Techniques: Classical Approaches and New Trends," *IET Communications*, 2007.
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE CVPR*, 2016.
5. M. Song, X. Song, and K. Qin, "Signal Detection and Demodulation Algorithm Based on Deep Learning in Communication Network," *3rd International Conference on Artificial Intelligence and Autonomous Robot Systems (AIARS)*, 2024.
6. Muthulakshmi S. and R. Jose, "Signal Demodulation without Channel Equalizer Using Machine Learning Techniques," *2nd International Conference on Intelligent Computing, Instrumentation and Control Technologies (ICICICT)*, 2019.
7. L. Hu, H. Jiang, R. Lu, and C. Liu, "Signal Classification in Real-time Based on SDR using Convolutional Neural Network," *IEEE 2nd International Conference on Information Technology, Big Data and Artificial Intelligence (ICIBA)*, 2021.
8. C. Gravelle and R. Zhou, "SDR Demonstration of Signal Classification in Real-Time using Deep Learning," *IEEE Globecom Workshops (GC Wkshps)*, 2019.
9. S. Y. Chaudhry and J. Haider, "Demonstration of a USRP-based Communication Network for Internet-of-Things (IoT) Application."

---

*This project was completed as part of a one-month research internship in the Department of Electronics and Communication Engineering, IIT Roorkee, under the supervision of Dr. Anshul Jaiswal.*
