<img width="1771" height="857" alt="image" src="https://github.com/user-attachments/assets/25d332b1-14ed-4a9b-8b0b-0908b18f4d1b" />

# Sentiment analysis system: A Compound AI Architecture

A production-ready, high-performance **Compound AI System** designed to decode informal digital consumer text, internet vernacular, and localized slang. 

This architecture balances computational efficiency with deep contextual awareness by running a fine-tuned, localized Transformer model optimized via **ONNX Runtime** alongside a reasoning **Agentic Layer powered by Gemini 1.5 Flash**.

---

## 🏗️ System Architecture

The application is structured as a dual-layer inference pipeline:
1. **The Core Compute Layer (Local):** A fine-tuned DistilBERT model optimized down to an inference latency of **~49 ms** using Hugging Face `optimum` and ONNX quantization. It handles rapid, cost-effective initial sentiment classification.
2. **The Agentic Layer (Cloud):** A Gemini 1.5 Flash LLM agent that acts as an intelligent triage system. When the local model encounters highly complex slang, regional dialects, or mixed contextual ambiguity, the agent is triggered to inject cultural semantics, explanation mapping, and validation.

Additionally, the dashboard embeds **SHAP (Shapley Additive Explanations)** to calculate exact token-level feature attribution heatmaps, offering full machine learning interpretability (XAI).

---

## 🚀 Key Engineering & MLOps Highlights

* **Green Tech / High Efficiency Quantization:** Converted baseline PyTorch weights into an **ONNX Runtime engine**, reducing production CPU latency by roughly **3x** (from ~150ms down to **49.37ms**), minimizing compute footprints and cloud operational costs.
* **Explainable AI (XAI):** Integrated game-theoretic SHAP mathematical heatmaps natively within the user interface to visualize exactly how individual slang tokens (e.g., "fire", "mid", "no cap") alter the model's output probabilities.
* **Enterprise Experiment Tracking:** Utilized **MLflow** as a centralized tracking metric store, prioritizing **$F1$-Score, Precision, and Recall** over simple accuracy to properly evaluate highly imbalanced social media datasets.
* **CI/CD Quality Assurance:** Configured a cross-platform (Windows/Linux) **GitHub Actions workflow** running automated testing modules via `pytest` to ensure robust data preprocessing pipelines.
* **Ethical AI & Dialect Governance:** Authored a comprehensive `MODEL_CARD.md` detailing dataset provenance and explicitly documenting model behavior against localized UK regional sub-cultures (e.g., Multicultural London English / Grime and Drill vernacular) in compliance with the UK AI Regulation Framework.

---

## 📂 Repository Structure

```text
├── .github/workflows/
│   └── python-app.yml       # CI/CD Automated Test Pipeline
├── app/
│   └── app.py               # Streamlit UI Dashboard & Visualization Layer
├── models/
│   └── onnx_model/          # Highly optimized ONNX Model weights & configs
├── src/
│   ├── agent.py             # Agentic Triage Layer (Gemini 1.5 Flash Integration)
│   ├── data_processor.py    # Structural Data Cleansing & Emoji Parser
│   ├── model_engine.py      # Core Inference Pipeline (ORTModel For Sequence Classification)
│   └── train_model.py       # Training Script with MLflow Tracking & F1 Metrics
├── Dockerfile               # Production multi-stage build script
├── .dockerignore            # Build optimization mask
├── .gitignore               # Secrets and heavy local DB masking
├── MODEL_CARD.md            # Regional dialect bias & model governance card
└── requirements.txt         # Pinned production dependency matrix

```

---

## 🛠️ Verification & Execution Guide

### 1. Prerequisites
Ensure you have Docker Desktop installed and running on your system, and have a valid Gemini API Key ready.

### 2. Standard Local Python Execution (Alternative)
To verify or run individual pipeline components outside a container:

```bash
# Install pinned dependencies
pip install -r requirements.txt

# Run the localized ONNX inference validation test
python src/model_engine.py

# Launch the MLflow UI server locally
mlflow ui
```

### 3. Containerized Production Launch (Docker)

To build the fully isolated, cross-platform container image and launch the system:

```bash
# Build the production docker image
docker build -t sentiment-agent-app:latest .

# Run the container isolated on Port 8501
# Inject your API key safely at runtime without embedding it in the image weights
docker run -d \
  -p 8501:8501 \
  -e GEMINI_API_KEY="YOUR_LIVE_GEMINI_API_KEY" \
  --name sentiment-prod-container \
  --restart always \
  sentiment-agent-app:latest
```


Once the container reports a healthy status, navigate to your local browser interface to interact with the system:

👉 http://localhost:8501

---

## 📊 Evaluation & Production Metrics Output

```text
INFO:src.model_engine:🚀 Loading optimized ONNX model from /app/models/onnx_model
INFO:src.model_engine:✅ ONNX pipeline loaded successfully with Optimum!
INFO:src.model_engine:⚡ ONNX Inference Latency: 49.37 ms for 1 text inputs.
Test Result: [{'label': 'POSITIVE', 'score': 0.8498}]
```
<img width="1167" height="327" alt="image" src="https://github.com/user-attachments/assets/76145d85-bfd1-47db-8eff-b199729086d1" />
<img width="1148" height="342" alt="image" src="https://github.com/user-attachments/assets/b777b65f-a16c-4a40-9528-b1a6b8bcf090" />

Advanced evaluation logs, precision, recall, and F1-curves are continuously tracked into the local mlruns database, visible instantly via your local MLflow server.
