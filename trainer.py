"""
S2MoE Qwen VL Trainer
Handles training loop, logging, and evaluation
"""

import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DefaultDataCollator,
)
import wandb
from datetime import datetime
import numpy as np
from pathlib import Path
from huggingface_hub import login

try:
    login(token=os.getenv("HF_TOKEN"))
except:
    print("⚠️ HuggingFace token not found, using existing CLI login...")

# Import local modules
from dataloader_qwen import QwenVLDataset
from dataset_qwen import load_llava_for_qwen
from S2MOE_LORA import S2MoE_LoRA_MLP, LoRA_MOE_LM
from utils import replace_mlp, collate_fn


class S2MoETrainer(Trainer):
    """Custom Trainer with S2MoE auxiliary loss support"""
    
    def __init__(self, *args, log_dir=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.config = config or {}
        self.step_logs = []
        self.expert_usage = []
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to include auxiliary losses"""
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute main loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        main_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Collect auxiliary losses and expert statistics
        aux_losses = {'Lb': [], 'Lu': []}
        expert_stats = {'gating_probs': [], 'selected_experts': [], 'entropy': []}
        alpha_bal = self.config.get("alpha_bal", 0.01)
        beta_unc = self.config.get("beta_unc", 0.1)

        
        for name, module in model.named_modules():
            if isinstance(module, (S2MoE_LoRA_MLP, LoRA_MOE_LM)):
                # Collect auxiliary losses
                if hasattr(module, '_aux_losses') and module._aux_losses:
                    if 'Lb' in module._aux_losses:
                        aux_losses['Lb'].append(module._aux_losses['Lb'])
                    if 'Lu' in module._aux_losses: # S2MoE only
                        aux_losses['Lu'].append(module._aux_losses['Lu'])
                
                # Collect expert usage statistics
                if hasattr(module, '_last_gating_probs'):
                    probs = module._last_gating_probs
                    expert_stats['gating_probs'].append(probs.mean(0).cpu().detach())
                    
                    # Calculate entropy of expert selection
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
                    expert_stats['entropy'].append(entropy.cpu().item())

                # Log noise statistics
                if isinstance(module, S2MoE_LoRA_MLP):
                    if hasattr(module, "_last_noise_std"):
                        wandb.log({
                            "train/noise_std_mean": module._last_noise_std.mean().item(),
                            "train/noise_mean_mean": module._last_noise_mean.mean().item(),
                            "train/merge_gate_mean": module._last_gate_value.mean().item(),
                        }, step=self.state.global_step)
        
        # Compute weighted auxiliary losses
        # total_loss = loss
        lb_loss_val = 0.0
        lu_loss_val = 0.0
        
        if aux_losses['Lb']:
            lb_loss = torch.stack(aux_losses['Lb']).mean()
            # total_loss = total_loss + alpha_bal * lb_loss
            lb_loss_val = lb_loss.item()
        else:
            lb_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        
        if aux_losses['Lu']:
            lu_loss = torch.stack(aux_losses['Lu']).mean()
            # total_loss = total_loss + beta_unc * lu_loss
            lu_loss_val = lu_loss.item()
        else:
            lu_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)

        total_loss = main_loss + alpha_bal * lb_loss + beta_unc * lu_loss
        

        # Log metrics
        step_log = {
            "step": self.state.global_step,
            "main_loss": main_loss.item(), # fair comparison
            "lb_loss": lb_loss_val,
            "lu_loss": lu_loss_val,
            "total_loss": total_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
        
        # Add expert statistics
        if expert_stats['entropy']:
            step_log["expert_entropy"] = np.mean(expert_stats['entropy'])
            step_log["expert_usage"] = [p.tolist() for p in expert_stats['gating_probs'][:5]]
        
        self.step_logs.append(step_log)
        
        # Log to wandb
        wandb.log({
            "train/main_loss": main_loss.item(),
            "train/perplexity": torch.exp(main_loss).item(),

            #auxiliary loss (s2moe only)
            "train/lb_loss": lb_loss_val,
            "train/lu_loss": lu_loss_val,
            "train/total_loss": total_loss.item(),
            "train/expert_entropy": step_log.get("expert_entropy", 0),
            "train/learning_rate": step_log["learning_rate"],
        }, step=self.state.global_step)
        
        inputs["labels"] = labels
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluation_loop(self, *args, **kwargs):
        """Override evaluation to collect detailed metrics"""
        output = super().evaluation_loop(*args, **kwargs)
        
        # Collect expert usage statistics during eval
        if hasattr(self, 'model'):
            expert_usage = []
            for name, module in self.model.named_modules():
                if isinstance(module, S2MoE_LoRA_MLP):
                    if hasattr(module, '_last_gating_probs'):
                        expert_usage.append(module._last_gating_probs.mean(0).cpu().detach().tolist())
            
            if expert_usage:
                self.expert_usage.append({
                    "step": self.state.global_step,
                    "expert_probs": expert_usage
                })
        
        return output


class DetailedLoggingCallback(TrainerCallback):
    """Callback for detailed logging"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')
        self.best_main_loss = float('inf') # track best main loss
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Save training logs
            if 'loss' in logs:
                self.train_losses.append({
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'loss': logs['loss'],
                })
            
            # Log to wandb
            wandb.log(logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Save eval metrics
            eval_log = {
                'step': state.global_step,
                'epoch': state.epoch,
                **metrics
            }
            self.eval_losses.append(eval_log)
            
            # Track best model
            eval_loss = metrics.get('eval_loss', float('inf'))
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                print(f"\n🏆 New best eval loss: {self.best_eval_loss:.4f} at step {state.global_step}")
            
            # Log to wandb
            # wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)
            # NEW: Log eval metrics with clear naming
            wandb.log({
                "eval/main_loss": eval_loss,  # 👈 FAIR metric for comparison
                "eval/perplexity": torch.exp(torch.tensor(eval_loss)).item(),
                **{f"eval/{k}": v for k, v in metrics.items()},
            }, step=state.global_step)
            
            # Save metrics to file
            with open(f"{self.log_dir}/eval_metrics.jsonl", "a") as f:
                f.write(json.dumps(eval_log) + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save all collected metrics
        with open(f"{self.log_dir}/train_losses.json", "w") as f:
            json.dump(self.train_losses, f, indent=2)
        
        with open(f"{self.log_dir}/eval_losses.json", "w") as f:
            json.dump(self.eval_losses, f, indent=2)

        # NEW: save best metrics summary
        best_metrics = {
            "best_eval_loss": self.best_eval_loss,
            "best_eval_perplexity": torch.exp(torch.tensor(self.best_eval_loss)).item(),
        }
        with open(f"{self.log_dir}/best_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
        
        print(f"\n💾 Saved training logs to {self.log_dir}")
        print(f" !! Best eval loss: {self.best_eval_loss:.4f}")
        print(f" !! Best eval perplexity: {torch.exp(torch.tensor(self.best_eval_loss)).item():.4f}")


class MemoryLoggingCallback(TrainerCallback):
    """Callback for GPU memory monitoring"""
    
    def on_step_end(self, args, state, control, **kwargs):


        if state.global_step % 100 == 0 and torch.cuda.is_available():
            memory_stats = {
                "memory/allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "memory/reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
                "memory/max_allocated_gb": torch.cuda.max_memory_allocated(0) / 1e9,
            }
            wandb.log(memory_stats, step=state.global_step)



def train_s2moe_model(base_dir, model_path, volume=None, subset_size=None):
    """
    Main training function for S2MoE Qwen VL model
    
    Args:
        base_dir: Base directory for data
        model_path: Path to pretrained model
        volume: Modal volume object (optional)
        subset_size: Use subset of data for testing (optional)
    
    Returns:
        training_summary: Dict with training results
    """

    # === FIX subset string - int
    if subset_size is not None:
        if isinstance(subset_size, str):
            subset_size = int(subset_size)
    
    # === Setup ===
    TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
    OUTPUT_DIR = f"{base_dir}/qwen_s2moe_finetuned_{TIMESTAMP}"
    LOGS_DIR = f"{OUTPUT_DIR}/logs"
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # === Hyperparameters ===
    config = {
        "model_name": "Qwen2.5-VL-7B",
        "architecture": "S2MoE-LoRA",
        "num_experts": 4,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.15,
        "top_k": 1,
        "alpha_bal": 0.01,
        "beta_unc": 0.1,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "warmup_ratio": 0.01,
        "max_length": 1024,
        "weight_decay": 0.01,
        "timestamp": TIMESTAMP,
    }
    
    # Save config
    with open(f"{LOGS_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    wandb.init(
        project="qwen-s2moe-finetuning",
        name=f"s2moe-training-{TIMESTAMP}",
        config=config,
        dir=LOGS_DIR,
    )

    print(f"Model path: {model_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Logs directory: {LOGS_DIR}")

    # === Load dataset ===
    print("\nLoading dataset...")
    hf_dataset = load_llava_for_qwen(base_dir)
    
    # Optional: Use subset for testing
    if subset_size:
        subset_size = min(subset_size, len(hf_dataset))
        hf_dataset = hf_dataset.select(range(subset_size))
        print(f" Using subset: {subset_size} samples for testing")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Split dataset
    train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = QwenVLDataset(train_test_split["train"], processor, max_length=config["max_length"])
    eval_dataset = QwenVLDataset(train_test_split["test"], processor, max_length=config["max_length"])
    
    dataset_info = {
        "total_samples": len(hf_dataset),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_ratio": len(train_dataset) / len(hf_dataset),
    }
    
    print(f"Dataset loaded:")
    print(f"   Total: {dataset_info['total_samples']:,}")
    print(f"   Train: {dataset_info['train_samples']:,}")
    print(f"   Eval: {dataset_info['eval_samples']:,}")
    
    # Save dataset info
    with open(f"{LOGS_DIR}/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    wandb.config.update(dataset_info)

    # === Load model ===
    print("\nLoading model...")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory before loading:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # === Freeze base parameters ===
    for p in model.parameters():
        p.requires_grad = False

    # === Replace MLPs with S2MoE_LoRA_MLP ===
    print("\nReplacing MLPs with S2MoE_LoRA_MLP...")
    replace_mlp(model, is_s2moe=True)

    # #=== Replace MLPs with MoLE ===
    # print("\nReplacing MLPs with MoLE...")
    # replace_mlp(model, is_s2moe=False)

    # codex debug
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # === Verify trainable parameters ===
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_params / total_params * 100
    
    model_info = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
        "frozen_params": total_params - trainable_params,
    }
    
    print(f"\nModel parameters:")
    print(f"   Total: {model_info['total_params']:,}")
    print(f"   Trainable: {model_info['trainable_params']:,} ({trainable_ratio:.2f}%)")
    print(f"   Frozen: {model_info['frozen_params']:,}")
    
    # Save model info
    with open(f"{LOGS_DIR}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    wandb.config.update(model_info)
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory after model load:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_dir=f"{LOGS_DIR}/trainer_logs",
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"s2moe-qwen-1k-{TIMESTAMP}",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=5.0,
        save_safetensors=True,
        push_to_hub=True,
        hub_model_id=f"s2moe-mole-qwen-finetuned1-{TIMESTAMP}",
        hub_strategy="end",

        #torch_compile=True,
        #torch_compile_backend="inductor",
        #torch_compile_mode="reduce-overhead",
    )


    # === Initialize Trainer ===
    trainer = S2MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: collate_fn(batch, processor),
        callbacks=[
            DetailedLoggingCallback(LOGS_DIR),
            MemoryLoggingCallback(),
        ],
        log_dir=LOGS_DIR,
        config=config,
    )

    # === Start Training ===
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Training config:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Steps per epoch: {len(train_dataset) // (config['batch_size'] * config['gradient_accumulation_steps'])}")
    print("="*60)
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        print("push to huggingface hub...")
        trainer.push_to_hub(commit_message="Final model upload")
        
        # Save step-level logs
        with open(f"{LOGS_DIR}/step_logs.json", "w") as f:
            json.dump(trainer.step_logs, f, indent=2)
        
        # Save expert usage
        with open(f"{LOGS_DIR}/expert_usage.json", "w") as f:
            json.dump(trainer.expert_usage, f, indent=2)
        
        # Log final metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        # NEW: main loss for fair comparison
        final_train_main_loss = metrics.get('train_loss', float('inf'))
        final_eval_main_loss = eval_metrics.get('eval_loss', float('inf'))
        
        # Save training summary
        training_summary = {
            "config": config,
            "dataset_info": dataset_info,
            "model_info": model_info,

            #  FAIR comparison metrics
            "final_metrics": {
                "train_main_loss": final_train_main_loss,
                "train_perplexity": torch.exp(torch.tensor(final_train_main_loss)).item(),
                "eval_main_loss": final_eval_main_loss,
                "eval_perplexity": torch.exp(torch.tensor(final_eval_main_loss)).item(),
            },

            "final_train_metrics": metrics,
            "final_eval_metrics": eval_metrics,
            "output_dir": OUTPUT_DIR,
            "timestamp": TIMESTAMP,
        }
        
        with open(f"{LOGS_DIR}/training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2)
        
        # Commit volume changes
        if volume:
            print("\nCommitting volume changes...")
            volume.commit()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"Model saved to: {OUTPUT_DIR}")
        print(f"Logs saved to: {LOGS_DIR}")
        print("="*60)
        
        wandb.finish()
        
        return training_summary
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        with open(f"{LOGS_DIR}/error.log", "w") as f:
            f.write(traceback.format_exc())
        
        wandb.finish(exit_code=1)
        raise

