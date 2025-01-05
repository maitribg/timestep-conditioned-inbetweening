import torch
from pathlib import Path
import mlflow
from src.eval import FrameInterpolationEvaluator

# Model imports
from src.models.Baseline1_OpticalFlow import OpticalFlow
from src.models.Baseline2_SlerpLatentDiffusion import SlerpLatentDiffusion
from src.models.Baseline3_KeyframeConditionedDiffusion import KeyframeConditionedDiffusion
from src.models.Experiment1_TimeEncodingDiffusion import TimeEncodingDiffusion
from src.models.Experiment2_WeightedCrossAttention import WeightedCrossAttention

def train_model(model_name, model, train_loader, epochs=5):
    """Simple training loop with MLflow logging"""
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", epochs)
        
        for epoch in range(epochs):
            print(f"Training {model_name} - Epoch {epoch+1}/{epochs}")
            
            # Training would happen here
            # For now, just log placeholder metrics
            loss = 0.5 - (epoch * 0.05)  # Simulated decreasing loss
            
            mlflow.log_metric("loss", loss, step=epoch)
        
        print(f"Completed training {model_name}")

def evaluate_and_log(model_name, results_dir):
    """Evaluate model and log metrics to MLflow"""
    
    evaluator = FrameInterpolationEvaluator(results_dir)
    all_results = evaluator.evaluate_all()
    stats = evaluator.compute_statistics(all_results)
    
    # Log evaluation metrics
    mlflow.log_metric("ssim", stats['ssim_overall_mean'])
    mlflow.log_metric("lpips", stats['lpips_overall_mean'])
    mlflow.log_metric("t_lpips", stats['t_lpips_mean'])
    mlflow.log_metric("t_of", stats['t_of_mean'])

if __name__ == "__main__":
    mlflow.set_experiment("animation-inbetweening")
    
    models = [
        "OpticalFlow",
        "SlerpLatentDiffusion", 
        "KeyframeConditionedDiffusion",
        "TimeEncodingDiffusion",
        "WeightedCrossAttention"
    ]
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Processing: {model_name}")
        print(f"{'='*50}\n")
        
        # Training would load actual model here
        # train_model(model_name, model, train_loader)