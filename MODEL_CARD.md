# Model Card: Slang-Aware Sentiment Engine

## 1. Model Details
- **Developer:** Aswathi PI
- **Model Architecture:** DistilBERT (Fine-tuned for Internet Slang and Modern Vernacular)
- **Optimization:** Exported to ONNX Runtime via Hugging Face `optimum` for high-efficiency CPU inference.
- **Language:** English (Primarily modern digital sociolects / Gen-Z vernacular).
- **License:** MIT

## 2. Intended Use
- **Primary Intended Use:** Rapid triage of informal consumer text, social media reviews, gaming chats, and brand mentions containing high-density slang and emojis.
- **Secondary Intended Use:** Serving as the fast processing layer in a Compound AI System, working alongside a Large Language Model (Gemini 1.5 Flash) agent that handles reasoning and deep context extraction.

## 3. Quantitative Performance
- **Baseline Model Latency:** ~120ms - 180ms (Standard PyTorch execution on CPU).
- **Optimized ONNX Latency:** ~45ms - 50ms (ONNX Runtime engine on CPU).
- **Performance Gain:** ~3x acceleration in production throughput, significantly reducing cloud operational costs and alignment with "Green AI" computing metrics.

## 4. Training Data & Provenance
- The local classifier was fine-tuned on a curated dataset of informal consumer reviews, text snippets, and social data emphasizing popular digital slang tokens (e.g., "fire", "mid", "banger", "no cap") paired with structural text cleansing via emoji-to-text token mappings.

## 5. Ethical Considerations & Dialect Bias (UK Context)
This model card explicitly acknowledges data limitations regarding socioeconomic and regional linguistic variations:
- **Geographic Bias:** The primary training sets are heavily weighted toward globalized Internet English and mainstream North American/Western digital terms.
- **UK Regional Dialect Nuances:** The model exhibits performance degradation when evaluating highly localized UK sociolects, such as **Multicultural London English (MLE)** or specific urban/regional vernaculars (e.g., regional drill/grime terminology). Words that convey high-praise or intense emphasis within these British sub-cultures may be misinterpreted as neutral or negative by the baseline DistilBERT tokenizer.
- **Mitigation Strategy:** Misclassifications and edge-case regional semantics are safely caught by an active **Agentic Triage Layer** using Gemini 1.5 Flash, which injects localized digital cultural awareness to correct the local model's decisions when confidence thresholds drop.

## 6. Document Governance
- **Compliance Alignment:** Aligned with the UK AI Regulation White Paper core principles: Safety, Security, and Robustness; Transparency and Explainability; Fairness.