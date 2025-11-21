from S2MOE_LORA import LoRALayer, LoRA_MOE_LM, S2MoE_LoRA_MLP, MlpWithLoRAMoE, MlpWithS2MoELoRA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

# ======================
# 4. Replace MLPs
# ======================
def replace_mlp(model, is_s2moe=False):
    num_layers = len(model.language_model.layers)    

    if(is_s2moe==False):
        print(f"Replacing {num_layers} MLPs with LoRA_MOE_LM...")
        for i in range(num_layers):
            orig_mlp = model.language_model.layers[i].mlp
            lora_moe_layer = LoRA_MOE_LM(orig_mlp, num_experts=4, rank=8, alpha=32, dense_moe=False)
            # Use wrapper to ensure HuggingFace compatibility when pushing to hub
            model.language_model.layers[i].mlp = MlpWithLoRAMoE(orig_mlp, lora_moe_layer).bfloat16()
    
        print(" MLPs replaced with LoRA_MOE_LM")
    else:
        print(f"Replacing {num_layers} MLPs with S2MoE_LoRA...")
        for i in range(num_layers):
            orig_mlp = model.language_model.layers[i].mlp
            s2moe_layer = S2MoE_LoRA_MLP(orig_mlp, num_experts=4, rank=8, alpha=32,
                                        lora_dropout=0.05, top_k=1,
                                        alpha_bal=0.01, beta_unc=0.1)
            # Use wrapper to ensure HuggingFace compatibility when pushing to hub
            model.language_model.layers[i].mlp = MlpWithS2MoELoRA(orig_mlp, s2moe_layer).bfloat16()

        print(" MLPs replaced with S2_LoRA")

    
    # ======================
    # 5. Check trainable ratio
    # ======================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params * 100
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Learnable ratio: {ratio:.2f}%")
    
    # ======================
    # 6. Print trainable layers
    # ======================
    print("\n=== Learnable parameters ===")
    for n, p in model.named_parameters():
        if p.requires_grad:
                print(f"{n:80s} {tuple(p.shape)}")


IGNORE_INDEX = -100

def collate_fn(batch, processor):
    """
    Collate function cho Qwen2.5-VL với pixel_values dạng flat embedding.
    
    Hỗ trợ:
    - Batch > 1
    - Pad text và vision embedding theo chiều dài lớn nhất
    - Mask phần prompt nếu có 'prompt_len'
    """

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Tách từng phần tử
    input_ids_list, attention_mask_list, labels_list = [], [], []
    pixel_values_list, image_grid_thw_list = [], []

    for sample in batch:
        # Text
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"].clone()

        # Mask phần prompt (nếu có)
        if "prompt_len" in sample:
            prompt_len = sample["prompt_len"]
            labels[:prompt_len] = IGNORE_INDEX

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

        # Vision embeddings
        pixel_values = sample["pixel_values"]
        assert pixel_values.dim() == 2, \
            f"Expected flat embedding (num_patches, dim), got {pixel_values.shape}"
        pixel_values_list.append(pixel_values)

        # Grid info
        if "image_grid_thw" in sample:
            image_grid_thw_list.append(sample["image_grid_thw"])


    # Pad text sequences
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)


    # Concatenate vision embeddings
    pixel_values = torch.cat(pixel_values_list, dim=0)  # (total_patches, dim)

    # Stack image grid (nếu có)
    if len(image_grid_thw_list) > 0:
        image_grid_thw = torch.stack(image_grid_thw_list, dim=0)
    else:
        image_grid_thw = None

    # Gộp thành batch
    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,   # (B, max_patches, hidden_dim)
    }

    if image_grid_thw is not None:
        batch_out["image_grid_thw"] = image_grid_thw

    return batch_out
