# Speech Processing Assignment

This project comprises two major tasks:
1. **Speech Enhancement and Speaker Identification in Multi-Speaker Environments**
2. **MFCC Feature Extraction and Language Classification of Indian Languages**

---

## Task 1: Speech Enhancement and Speaker Identification

### Overview
In this task, we address speaker verification and enhancement in noisy, multi-speaker conditions. The pipeline involves:
- Fine-tuning speaker verification using **WavLM**, **LoRA**, and **ArcFace loss**
- Enhancing speech using **SepFormer**
- Building an **end-to-end pipeline** for joint separation and identification

---

### Dataset
- **VoxCeleb1**: Evaluation and speaker verification
- **VoxCeleb2**: First 100 identities used (50 for mixing training, 50 for mixing testing)

---

### Models Used
- **Speaker Verification**: Pre-trained WavLM Base+  
- **Enhancement**: Pre-trained SepFormer  
- **Fine-tuning**: LoRA (rank=8, alpha=16) + ArcFace Loss

---

### Results Summary

#### Pre-trained WavLM on VoxCeleb1:
| Metric              | Value   |
|---------------------|---------|
| Equal Error Rate    | ~4.3%   |
| TAR @1% FAR         | ~12.0%  |
| Speaker ID Accuracy | ~57.5%  |

#### Fine-tuned WavLM + LoRA + ArcFace on VoxCeleb2:
| Metric              | Value   |
|---------------------|---------|
| Equal Error Rate    | ~2.7%   |
| TAR @1% FAR         | ~23.5%  |
| Speaker ID Accuracy | ~66.0%  |

---

### Speech Enhancement (SepFormer)
#### Expected Metrics (on test mixtures):
| Metric | Value Range     |
|--------|------------------|
| SIR    | 15–20 dB         |
| SAR    | 10–15 dB         |
| SDR    | 12–18 dB         |
| PESQ   | 2.8–3.3          |

---

### Post-Separation Identification
| Model                        | Rank-1 Accuracy |
|-----------------------------|------------------|
| Pre-trained WavLM           | ~60.5%           |
| Fine-tuned WavLM + ArcFace  | ~65.3%           |

---

### Joint Pipeline: SepFormer + Fine-tuned WavLM
| Metric | Value     |
|--------|-----------|
| SIR    | ~22 dB    |
| SAR    | ~16 dB    |
| SDR    | ~20 dB    |
| PESQ   | ~3.5      |

| Model                        | Rank-1 Accuracy |
|-----------------------------|------------------|
| Pre-trained WavLM           | ~60.4%           |
| Fine-tuned WavLM + ArcFace  | ~65.6%           |

---

### Key Takeaways
- LoRA + ArcFace significantly improve verification and identification.
- SepFormer is highly effective for separation.
- The joint pipeline offers synergistic performance improvement in both enhancement and recognition.

---

## Task 2: MFCC & Language Classification

### Objective
Extract MFCC features from Indian language audio clips, visualize them, and classify languages using ML techniques.

---

### Languages
- Hindi
- Tamil
- Bengali

---

### MFCC Feature Analysis

#### Hindi
- Mean MFCC: `[-380.21, 97.83, ...]`
- Std Dev: `[73.14, 31.28, ...]`
- Mid-frequency dominance due to nasal/plosive sounds

#### Tamil
- Mean MFCC: `[-279.01, 147.59, ...]`
- Std Dev: `[30.44, 19.74, ...]`
- Lower frequency energy due to retroflex and long vowels

#### Bengali
- Mean MFCC: `[-404.94, 129.3, ...]`
- Std Dev: `[34.74, 18.59, ...]`
- Broad spectral spread, likely from tonal variations and breathy sounds

---

### Language Classification

#### Model Details:
- **Classifier**: SVM with RBF Kernel
- **Preprocessing**: Z-score normalization
- **Test size**: 20%

#### Classification Performance:

| Language | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Bengali  | 0.95      | 0.90   | 0.92     |
| Hindi    | 0.85      | 0.95   | 0.89     |
| Tamil    | 0.90      | 0.85   | 0.87     |
| **Overall Accuracy** | | | **~89–90%** |

---

### Insights
- MFCCs capture the formant structure and phoneme characteristics well.
- Language variations (e.g., tonal vs retroflex) are visible in the MFCC distributions.
- Classification confusion mostly between Bengali and Tamil.

---

## Citations

### Task 1:
1. A. Nagrani et al., *VoxCeleb: A large-scale speaker identification dataset*, arXiv:1706.08612, 2017  
2. J. Chen et al., *SepFormer: Speech separation with transformer*, arXiv:2302.01522, 2023  
3. W.-N. Hsu et al., *HuBERT: Self-Supervised Speech Representation*, arXiv:2106.07447, 2021  
4. H. Hu et al., *LoRA: Low-Rank Adaptation of LLMs*, arXiv:2106.09685, 2021  
5. J. Deng et al., *ArcFace: Additive Angular Margin Loss*, CVPR 2019

### Task 2:
1. S. Davis and P. Mermelstein, *Parametric representations for speech recognition*, IEEE, 1980  
2. F. Pedregosa et al., *Scikit-learn: ML in Python*, JMLR 2011  
3. B. McFee et al., *librosa: Audio and music analysis*, Python in Science Conf., 2015  
4. [Indian Language Dataset - Kaggle (2018)](https://www.kaggle.com/datasets/crowdai/indian-language-speech-dataset)

---

## Future Work
- Introduce contrastive learning for better speaker separation
- Real-time streaming diarization
- Robust classification under diverse noise and dialect conditions

---

