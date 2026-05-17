import os
import logging
import time
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentEngine:
    def __init__(self, model_path=None):
        """
        Initializes the Optimized ONNX Sentiment Engine.
        """
        # Explicitly targets your local compiled binary graph path folder
        self.model_path = model_path or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "onnx_model"
        ))
        self.pipeline = None 

    def load_inference_pipeline(self):
        """
        Loads the quantized ONNX model and tokenizer into a Hugging Face pipeline.
        """
        try:
            # 1. Fallback safety step if the ONNX graph files are missing or incomplete
            if not os.path.exists(self.model_path) or not os.path.exists(os.path.join(self.model_path, "model.onnx")):
                logger.warning(f"⚠️ ONNX model not found at {self.model_path}. Loading standard fallback.")
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name)
            
            else:
                logger.info(f"🚀 Loading optimized ONNX model from {self.model_path}")
                
                # 2. Load the optimized structural graph model from your local disk path
                model = ORTModelForSequenceClassification.from_pretrained(
                    self.model_path, 
                    file_name="model.onnx"
                )
                
                # 3. ABSOLUTE FIXED SOLUTION: Hardcode a reliable model hub reference string 
                # to prevent AutoTokenizer from looking for config files in the binary folder!
                hub_reference = "distilbert-base-uncased-finetuned-sst-2-english"
                tokenizer = AutoTokenizer.from_pretrained(hub_reference)
                
                # 4. Bind the components into your classification pipeline
                self.pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer
                )
                logger.info("✅ ONNX pipeline loaded successfully with Optimum!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimized model: {str(e)}")
            raise e

    def predict_batch(self, texts):
        """
        Runs inference on a list of strings and logs the latency performance.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call load_inference_pipeline() first.")
        
        try:
            start_time = time.perf_counter()
            results = self.pipeline(texts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"⚡ ONNX Inference Latency: {latency_ms:.2f} ms for {len(texts)} text inputs.")
            
            return results
        except Exception as e:
            logger.error(f"❌ Inference error: {str(e)}")
            return [{"label": "ERROR", "score": 0.0}]

if __name__ == "__main__":
    # Internal baseline check
    engine = SentimentEngine()
    engine.load_inference_pipeline()
    test_text = "This optimization is absolute fire no cap!"
    print(f"Test Result: {engine.predict_batch([test_text])}")