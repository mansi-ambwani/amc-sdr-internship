# Automatic Modulation Classification using Deep Learning and a Software-Defined Radio Pipeline

Real-time classification of 8 PSK/QAM modulation schemes, built end-to-end during a one-month research internship at the Department of Electronics and Communication Engineering, IIT Roorkee (Supervisor: Dr. Anshul Jaiswal).

The project spans three tightly coupled pieces:
1. **Over-the-air dataset generation** — a full USRP transmit/receive pipeline (framing, RRC pulse shaping, correlation-based sync, CFO correction, channel equalization, Costas-loop phase tracking) implemented from scratch for 8 modulation schemes.
2. **RadioResNet95** — a 1-D residual CNN trained to classify 1024-sample I/Q windows into one of 8 modulation classes.
3. **Real-time integration** — a script that ties the live USRP receive chain directly to the trained model for on-the-fly classification.

---

## Results at a glance

| Test | Setup | Accuracy |
|---|---|---|
| Offline, held-out split | Trained on captured dataset (1,168 windows, 146/class) | **99.70%** (330/331 correct) |
| Live, over-the-air | Full USRP hardware chain + RadioResNet95, 10 frames/modulation | **63.75%** |

The entire PSK family (BPSK, QPSK, 8-PSK, 16-PSK) and 32-QAM classify reliably live. The three square, Gray-coded QAM orders — 16-QAM, 64-QAM, and 256-QAM — are frequently confused with one another in live testing despite the near-perfect offline result. **The root cause of this gap was not conclusively identified within the internship timeframe** — see [Known Limitations](#known-limitations--future-work) below. A candidate fix (widening the receiver's gain-estimation window) is implemented in some of the experiment branches under `models/`, but was not independently verified to resolve the issue.

Full methodology, derivations, and analysis are in [`report/`](./report).

---

## Repository structure

```
AMC-Deep-Learning-USRP/
│
├── README.md                              # Project overview, results table, how to run, links to Release
├── requirements.txt                       # Python deps: torch, numpy, scipy, matplotlib, seaborn, uhd
│
├── report/
│   └── AMC_Internship_Report_MansiAmbwani.docx
│                                           # Full write-up: theory, derivations, hardware pipeline,
│                                           # model architecture, results, challenges, future work
│
├── hardware_pipeline/                     # USRP TX/RX signal chain — one script per modulation.
│   │                                       # Each does: frame build (preamble+sync+payload) → RRC
│   │                                       # pulse shaping → UHD TX/RX over the air → correlation
│   │                                       # sync → CFO correction → channel equalization → Costas-
│   │                                       # loop phase tracking → save synchronized I/Q as .npy
│   ├── bpsk.py
│   ├── qpsk.py
│   ├── 8psk.py
│   ├── 16psk.py
│   ├── 16qam.py
│   ├── 32qam.py
│   ├── 64qam.py
│   └── 256qam.py
│
├── dataset/
│   ├── README.md                          # Explains what each file contains (shape, units, how it
│   │                                       # was generated) + download link to the Release asset
│   ├── ml_bpsk_samples.npy                 # ⚠️ UNBALANCED dataset — the original small capture:
│   ├── ml_qpsk_samples.npy                 #    ~146 windows/class (1,168 total), one .npy per
│   ├── ml_8psk_samples.npy                 #    modulation, output directly from the corresponding
│   ├── ml_16psk_samples.npy                #    hardware_pipeline/ script. This is what RadioResNet95
│   ├── ml_16qam_samples.npy                #    was actually trained/evaluated on in report/ Sections
│   ├── ml_32qam_samples.npy                #    5–7 (the 99.70% offline / 63.75% live results).
│   ├── ml_64qam_samples.npy
│   ├── ml_256qam_samples.npy
│   └── 1.5all_data.zip                    # ✅ BALANCED dataset — a separate, much larger capture:
│                                           #    1.5 lakh (150,000) windows per modulation, evenly
│                                           #    across all 8 classes. Used only for the supplementary
│                                           #    offline evaluation in report/ Section 7.3 (it was not
│                                           #    re-run through the live hardware chain). Too large for
│                                           #    a normal commit — upload as a GitHub Release asset,
│                                           #    keep only this filename + link here
│
├── models/                                # Four experiment variants, all sharing the RadioResNet95
│   │                                       # backbone (1D residual CNN, ~3.9M params, see below),
│   │                                       # differing only in the input-feature strategy
│   │
│   ├── best_model_8_modulations/          # Baseline: trained on raw I/Q windows only.
│   │   ├── model12.py                     #   Real-time USRP test harness — loads weights, runs
│   │   │                                   #   live capture → classification → confusion matrix
│   │   ├── Untitled17.ipynb               #   Training notebook: dataset load, model def, train
│   │   │                                   #   loop, offline evaluation
│   │   └── suyash_resnet_weights.pth      #   Trained weights — move to Release, link in README
│   │
│   ├── model_with_HOC/                    # + higher-order-cumulant statistical features appended
│   │   ├── hoc_1st.py                     #   to raw I/Q, same real-time test harness role as above
│   │   ├── hoc_1st.ipynb                  #   Training notebook for this variant
│   │   └── hoc_1st.pth                    #   Trained weights — move to Release
│   │
│   ├── model_with_overlapping/            # + overlapping (sliding) window sampling — generates
│   │   ├── 1st+overlap.py                 #   more training windows from the same raw capture
│   │   ├── 1st+overlap.ipynb
│   │   └── 1st.pth                        #   Trained weights — move to Release
│   │
│   └── model_with_hoc_overlapping/        # Combines both of the above techniques
│       ├── overlap+hoc_n.py
│       ├── overlap+hoc_n.ipynb
│       └── overlap+hoc_n.pth              #   Trained weights — move to Release
│
└── results/                                # Confusion-matrix images pulled out of each model's zip,
    │                                        # referenced directly in this README and report/ for quick
    │                                        # visual comparison without opening any notebook
    ├── offline_confusion_matrix.png        #   99.70% offline result (report Section 7.1)
    ├── usrp_ml_confusion_matrix.png                    # live result, best_model_8_modulations (63.75%)
    ├── realtime_8classhoc_confusion_matrix_v8.png      # live result, model_with_HOC
    ├── realtime_8class_confusion_matrix_v8.png         # live result, model_with_overlapping
    └── realtime_8class_confusion_matrix_v10.png        # live result, model_with_hoc_overlapping
```

**Why the `.pth`/`.zip` files are marked "move to Release":** GitHub's web uploader caps a browser commit at ~25MB total, and even git-based pushes shouldn't really carry 15MB weight files per variant × 4 model variants — every future `git clone` re-downloads all of them forever, and binaries like these can't be diffed like code anyway. A GitHub Release lets you attach up to 2GB per file, completely separate from commit history — see [Data & Model Weights](#data--model-weights) below.

> **Note on the current upload:** the repo right now has everything uploaded flat at the root (scripts, zips, `.npy` files together, some duplicated). This README describes the target structure above — see [Reorganizing the repo](#reorganizing-the-repo) for the exact commands to get there.

---

## Data & model weights

The raw dataset (`1.5all_data.zip`), the per-modulation `.npy` sample files, and trained weights (`.pth`, bundled inside each `models/*.zip`) are **not meant to live directly in git history** — they're multi-tens-of-MB binaries that bloat every future clone and can't be diffed anyway.

Instead:
- Code, notebooks, and result images are committed normally.
- Datasets and `.pth` weights are attached to a **[GitHub Release](../../releases)** and linked here once published.

**Two separate datasets — don't confuse them:**

| Dataset | Files | Size | Balance | Used for |
|---|---|---|---|---|
| Original capture | `dataset/ml_*_samples.npy` (one per modulation) | ~146 windows/class, 1,168 total | **Unbalanced** across classes | All headline results in `report/` (99.70% offline, 63.75% live) |
| Supplementary capture | `dataset/1.5all_data.zip` | 1.5 lakh (150,000) windows per modulation, 1.2M total | **Balanced** — equal count for all 8 classes | Offline-only supplementary evaluation, `report/` Section 7.3; not re-run through the live hardware chain |

| Asset | Description | Size |
|---|---|---|
| `1.5all_data.zip` | Balanced dataset, 150,000 windows/class across all 8 modulations | ~large, see Release |
| `best_model_8_modulations.pth` | Baseline RadioResNet95 weights | ~15 MB |
| `model_with_HOC.pth` | + higher-order-cumulant input features | ~15 MB |
| `model_with_overlapping.pth` | + overlapping window sampling | ~15 MB |
| `model_with_hoc_overlapping.pth` | + both combined | ~15 MB |

*(Links go live once you publish the Release — see the checklist below.)*

---

## The signal chain (`hardware_pipeline/`)

Each `<modulation>.py` script implements the full TX→RX loop for one modulation scheme over a matched pair of USRP radios via UHD:

1. **Frame construction** — a repeated CFO-estimation preamble + Barker-like sync word + Gray-coded payload symbols, pulse-shaped with a root-raised-cosine filter (β = 0.8, span = 6 symbols, 4 samples/symbol).
2. **Hardware TX/RX** — synchronized transmit and receive threads over UHD at a 1.5 GHz carrier, 20 MHz sample rate.
3. **Receiver DSP**:
   - Correlation-based frame/timing synchronization against the known preamble.
   - Carrier frequency offset (CFO) estimation and correction via a delay-and-correlate estimator.
   - Data-aided least-squares channel gain estimation and equalization.
   - Second-order Costas-loop carrier phase tracking with a decision-directed slicer, including M-fold phase-ambiguity resolution.
4. **Payload extraction** — synchronized, phase-tracked symbols are split into I/Q, reshaped, and saved as `.npy` windows ready for the CNN.

The receive chain was independently validated with near-zero Bit Error Rate across nearly all 8 schemes, confirming the physical link itself is reliable (see `report/` Section 4 & 8 for full derivations).

---

## The model (`models/`)

**RadioResNet95** — a 1-D residual CNN, ~3.9M trainable parameters:

- **Front-end**: a 2-D convolution with a (2×7) kernel collapsing the I/Q rows into 64 joint I/Q feature channels in one step.
- **Residual tower**: 4 stages × 2 residual blocks each, channel width doubling and temporal resolution halving per stage (64 → 128 → 256 → 512).
- **Head**: global average pooling → fully connected (512 → 128 → 8) with batch norm, ReLU, dropout (p = 0.3).
- **Training**: AdamW (η = 1e-3, weight decay 1e-4), cosine-annealed LR over 40 epochs, batch size 32, 80/20 stratified split, cross-entropy loss.

Four experiment variants are included, exploring different input-feature strategies on top of the same backbone:

| Folder | What it adds |
|---|---|
| `best_model_8_modulations/` | Baseline: raw I/Q windows only |
| `model_with_HOC/` | + higher-order-cumulant statistical features |
| `model_with_overlapping/` | + overlapping (sliding) window sampling for more training examples |
| `model_with_hoc_overlapping/` | Both of the above combined |

Each folder's notebook contains the full train/eval loop and its own confusion matrix. `best_model_8_modulations/` is the version referenced throughout `report/`.

---

## Running this yourself

```bash
pip install -r requirements.txt
# UHD/USRP hardware + drivers are required for the hardware_pipeline/ and
# real-time inference scripts; the models/ notebooks can be run standalone
# once you have the dataset (see Data & Model Weights above).
```

1. **Capture data**: run the relevant script in `hardware_pipeline/` for each modulation (requires a matched USRP TX/RX pair).
2. **Train**: open any notebook under `models/` and point it at your `.npy` windows.
3. **Real-time test**: `models/best_model_8_modulations/model12.py` loads trained weights and runs live USRP capture → classification → confusion matrix, exactly as described in `report/` Section 8.

---

## Reorganizing the repo

If your repo currently has everything flat at the root, run this locally (clone first) to match the structure above, then push once:

```bash
git clone https://github.com/mansi-ambwani/<your-repo>.git
cd <your-repo>

mkdir -p report hardware_pipeline dataset results \
         models/best_model_8_modulations models/model_with_HOC \
         models/model_with_overlapping models/model_with_hoc_overlapping

# code -> hardware_pipeline/
git mv bpsk.py qpsk.py 8psk.py 16psk.py 16qam.py 32qam.py 64qam.py 256qam.py hardware_pipeline/
git rm hardware_pipeline.zip        # redundant once the .py files are unzipped and moved

# report
git mv AMC_report_MansiAmbwani.docx report/

# dataset
git mv ml_*.npy dataset/
git mv 1.5all_data.zip dataset/     # then move to a Release and delete from git (see below)

# unzip each model archive locally, split code/notebook from weights:
#   - code (.py) + notebook (.ipynb) -> models/<variant>/
#   - confusion matrix .png          -> results/
#   - weights (.pth)                 -> upload to a Release, then remove from the repo

git add -A
git commit -m "Reorganize repo into report/, hardware_pipeline/, dataset/, models/, results/"
git push
```

Then, on GitHub: **Releases → Draft a new release**, attach `1.5all_data.zip` and the four `.pth` weight files, publish, and paste the resulting asset links into `dataset/README.md` and the [Data & Model Weights](#data--model-weights) table above.

---

## Known limitations & future work

- **Square-QAM confusion is unresolved.** 16-QAM, 64-QAM, and 256-QAM are frequently confused with each other live despite 99.70% offline accuracy. The receiver's payload gain-normalization stage is a candidate contributing factor (see `report/` Section 9), but this was not conclusively isolated as the root cause within the internship timeframe.
- **Small, unbalanced original dataset.** The headline results come from `dataset/ml_*_samples.npy` — ~146 windows/class (1,168 total), unbalanced across the 8 classes. A separate, balanced dataset (`dataset/1.5all_data.zip`, 150,000 windows/class) was assembled later but only used for supplementary offline evaluation — it was not re-run through the live hardware chain, so it doesn't confirm or refute the live square-QAM confusion.
- **Next steps**: controlled ablation experiments isolating the gain-estimation window (and other candidate factors) from the rest of the receive chain; re-running the live test on the balanced-dataset model; evaluating across a range of SNR levels.

Full discussion: `report/AMC_Internship_Report_MansiAmbwani.docx`, Sections 9–10.

---

## References

1. T. J. O'Shea, T. Roy, and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE Journal of Selected Topics in Signal Processing*, 2018. [doi:10.1109/JSTSP.2018.2797022](https://doi.org/10.1109/JSTSP.2018.2797022)
2. T. J. O'Shea and N. West, "Radio Machine Learning Dataset Generation with GNU Radio," *Proceedings of the GNU Radio Conference*, 2016. [PDF](https://pubs.gnuradio.org/index.php/grcon/article/view/11)
3. O. A. Dobre, A. Abdi, Y. Bar-Ness, and W. Su, "Survey of Automatic Modulation Classification Techniques: Classical Approaches and New Trends," *IET Communications*, 2007. [doi:10.1049/iet-com:20050176](https://doi.org/10.1049/iet-com:20050176)
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
5. M. Song, X. Song, and K. Qin, "Signal Detection and Demodulation Algorithm Based on Deep Learning in Communication Network," *AIARS*, 2024. [doi:10.1109/AIARS63200.2024.00091](https://doi.org/10.1109/AIARS63200.2024.00091)
6. Muthulakshmi S. and R. Jose, "Signal Demodulation without Channel Equalizer Using Machine Learning Techniques," *ICICICT*, 2019.
7. L. Hu, H. Jiang, R. Lu, and C. Liu, "Signal Classification in Real-time Based on SDR using Convolutional Neural Network," *ICIBA*, 2021. [doi:10.1109/ICIBA52610.2021.9687958](https://doi.org/10.1109/ICIBA52610.2021.9687958)
8. C. Gravelle and R. Zhou, "SDR Demonstration of Signal Classification in Real-Time using Deep Learning," *IEEE Globecom Workshops*, 2019.
9. S. Y. Chaudhry and J. Haider, "Demonstration of a USRP-based Communication Network for Internet-of-Things (IoT) Application," *36th Conference of FRUCT Association*.

---

## Author

**Mansi Ambwani** — Research Internship, Dept. of Electronics and Communication Engineering, IIT Roorkee
Supervisor: Dr. Anshul Jaiswal
