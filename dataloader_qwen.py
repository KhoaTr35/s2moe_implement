# dataloader_qwen.py
from torch.utils.data import Dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

class QwenVLDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=512):
        self.ds = hf_dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        # Qwen2.5-VL sử dụng chat template với messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item["query"].replace("<image>\n", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["response"]},
                ],
            }
        ]

        # Áp dụng chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Process image và text
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False, # không pad
            truncation=False, # tắt truncation
            #max_length=self.max_length,
            return_tensors="pt",
        )

        # Manual truncation nếu quá dài
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        if len(input_ids) > self.max_length:
            # Truncate from the end, keep image tokens
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Tạo labels (mask out input tokens, chỉ giữ response)
        labels = input_ids.clone()
        # Có thể cần mask phần input để chỉ train trên response
        
        inputs["labels"] = labels
        
        # return {k: v.squeeze(0) for k, v in inputs.items()}
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0),
            'labels': labels,
        }