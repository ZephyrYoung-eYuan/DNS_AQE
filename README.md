# Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models

**ICML 2026**

---

## Abstract

Diffusion models deliver state-of-the-art image quality but are expensive to deploy. Post-training quantization (PTQ) can shrink models and speed up inference, yet residual quantization errors distort the diffusion distribution (the timestep-wise marginal over $\mathbf{x}_t$), degrading sample quality. We propose a distribution-preserving framework that absorbs quantization error into the generative process without changing architecture or adding steps. Deformable Noise Scheduler (DNS) reinterprets quantization as a principled timestep shift, mapping the quantized prediction distribution $\mathbf{x}_t$ back onto the original diffusion distribution so that the target marginal is preserved. Unlike trajectory-preserving or noise-injection methods limited to stochastic samplers, our approach preserves the distribution under both stochastic and deterministic samplers and extends to flow-matching with Gaussian conditional paths. It is plug-and-play and complements existing PTQ schemes. Empirically, our method consistently enhances generation quality across diverse backbones and existing PTQ baselines. Notably, when further quantizing the FP16 LoRA branch of SVDQuant to enable fully integer inference, our approach effectively mitigates the performance drop, reducing FID from 27.16 to 26.22.
Code is available at https://github.com/ZephyrYoung-eYuan/DNS_AQE

---

## Usage

### 1. Install PTQD and Complete LDM Quantization

Follow the official instructions at [https://github.com/ziplab/PTQD](https://github.com/ziplab/PTQD) to set up the PTQD environment and complete the LDM quantization. This will produce quantized model checkpoints (e.g., `quantw4a8_ldm_brecq.pth`).

### 2. Integrate DNS into PTQD

After the PTQD quantization is done, copy the DNS source files into the PTQD project directory:

```bash
# Copy the main calibration & inference entry script
cp -r DNS_AQE/PTQD/quant_scripts/*  PTQD/quant_scripts/

# Copy the DNS core module
cp -r DNS_AQE/quant  PTQD/
```

### 3. Run DNS Calibration and Sampling

Execute the following command inside the PTQD project root to perform DNS calibration and generate corrected samples:

```bash
python quant_scripts/main_dns_ldm.py --cali --test
```

### 4. Evaluate with FID

To compute FID scores, use the evaluation script from OpenAI's guided-diffusion:

[https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py)

