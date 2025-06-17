# D-MoLE: Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning

<p align="center">
  <a href="https://arxiv.org/abs/2506.11672">
    <img src="https://img.shields.io/badge/ICML_2025-Paper-blue?style=for-the-badge&logo=readthedocs" alt="ICML 2025 Paper">
  </a>
</p>

Official implementation of our **ICML 2025** paper:  
**"Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning"**  
Feel free to star ⭐ this repo if you find it helpful!

---

## 🚀 Quick Start

### 1. Create Environment
```bash
conda create -n D-MoLE python=3.10 -y
conda activate D-MoLE
```

### 2. Install Dependencies
```bash
# Install uv package manager
pip install uv

# Install project in development mode
uv pip install -e .
```

### 3. Install Flash Attention
```bash
# Install flash-attention (pre-compiled wheel for CUDA 12 + PyTorch 2.6)
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Install Other Requirements
```bash
uv pip install opencv-python imageio decord pycocoevalcap wandb datasets
conda install openjdk=8 -y  # for pycocoevalcap
```

---

## 📁 Data Preparation

### Download Datasets

📥 **[Download from Google Drive](https://drive.google.com/file/d/1Ai9u3rwNctyMF8k3v3N3AwGR5ax6pAjD/view?usp=drive_link)**

1. **Download datasets** from the Google Drive link above and organize them in the following structure:
```
data/
├── flickr30k/
│   ├── Images/
│   ├── flickr30k_test_karpathy.jsonl
│   └── train.jsonl
├── vizwiz/
├── textcaps/
├── iconqa/
├── ocrvqa/
├── kvqa/
├── pmcvqa/
└── skvg/
```

2. **Download pretrained model** and place it at:
```
pretrained/
└── InternVL2-2B/
```

---

## 🧪 Run Experiments

We provide the complete D-MoLE training and evaluation pipeline.

### Step 1: Preprocessing
```bash
# Compute sequence representations and train autoencoder
bash scripts/preprocess/compute_seq_rep.sh
python scripts/preprocess/train_autoencoder.py
```

### Step 2: Architecture Search
```bash
# Compute zero-cost proxy scores and generate D-MoLE architecture
bash scripts/preprocess/compute_zc_score.sh
python scripts/preprocess/get_dmole_arch.py
```

### Step 3: Training
```bash
# Train D-MoLE model
bash scripts/train/dmole.sh

# Baseline: Sequential LoRA fine-tuning
bash scripts/train/seq_lora.sh
```

### Step 4: Evaluation
```bash
# Evaluation is enabled by default in the training scripts
# Or you can manually run evaluation using:
bash run_eval.sh
# This supports both pretrained mode and continual mode
```

You can modify these scripts to customize the base model, tasks, and other configurations.

---

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects:

- [InternVL](https://github.com/OpenGVLab/InternVL): Foundation multimodal model and training framework
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning library

We thank the authors and contributors of these projects for their valuable work.

---

## 📄 Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{ge2025dynamic,
  title     = {Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning},
  author    = {Chendi Ge and Xin Wang and Zeyang Zhang and Hong Chen and Jiapei Fan and Longtao Huang and Hui Xue and Wenwu Zhu},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {ICML '25},
  year      = {2025},
  publisher = {PMLR}
}
```

---

## 📬 Contact

If you have any questions, feel free to open an issue or contact the first author at `gcd23@mails.tsinghua.edu.cn`.

---

## 🪪 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.