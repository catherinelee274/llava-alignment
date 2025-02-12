import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset

VLFEEDBACKDATASET = "MMInstruction/VLFeedback"

dataset_keys=['MMInstruction/VLFeedback', 'MLLM-DPO']



class VQADataset(Dataset):
    def __init__(self, dataset, dataset_name, max_length, processor, evaluate_llm=False, images_root=None):
        self.dataset = dataset
        self.processor = processor
        self.evaluate_llm = evaluate_llm
        self.max_length = max_length
        self.images_root = images_root
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        rejected = None
        
        if self.dataset_name == dataset_keys[1]:
            images = [Image.open(self.images_root+example['image']).convert("RGB") for example in all_data]
            answers = [example['chosen']['value'] for example in all_data]
            rejected = [example['rejected']['value'] for example in all_data]
            prompts = [example['question']['value'] for example in all_data]
            encoding = self.processor(images, prompts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")

            #combine rejected and answers into processor to make sure have same length in batch       
            all_labels = []
            all_labels.extend(answers)
            all_labels.extend(rejected)
            all_labels = self.processor(text=all_labels, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
            labels = torch.LongTensor(all_labels[:len(answers)])
            neg_labels = torch.LongTensor(all_labels[len(answers):])

        else:        
            images = [Image.open(example['img_path']).convert("RGB") for example in all_data]
            answers = [example['completions']['response'][0] for example in all_data]
            prompts = [example['prompt'] for example in all_data]
            encoding = self.processor(images, prompts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
            labels = self.processor(text=answers, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids 
            labels = torch.LongTensor(labels) 
            neg_labels = None

        true_answers = self.processor.batch_decode(labels, skip_special_tokens=True, max_length=self.max_length)
        if self.evaluate_llm:
            questions = prompts
        else:
            questions = None

        input_ids = torch.LongTensor(encoding['input_ids'])
        pixel_values = torch.FloatTensor(encoding['pixel_values'])
        
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
                'input_ids':  input_ids,
                'pixel_values':   pixel_values,
                'attention_mask':  attention_mask,
                'labels': labels,
                'neg_labels': neg_labels,  #neg_labels,   #create from rejected
                'questions': questions,  
                'true_answers': true_answers
            }

        return batched_data
    

def get_vlfeedback_dataloaders(vqa_dataset, train_split, eval_split, processor, max_length, evaluate_llm=False):
    raw_train_dataset = load_dataset(vqa_dataset,split=train_split,trust_remote_code=True)
    raw_eval_dataset = load_dataset(vqa_dataset,split=eval_split,trust_remote_code=True)
    train_dataset = VQADataset(raw_train_dataset, vqa_dataset, max_length, processor)   #specific to "MMInstruction/VLFeedback"
    eval_dataset = VQADataset(raw_eval_dataset, vqa_dataset, max_length, processor, evaluate_llm=evaluate_llm)     #specific to "MMInstruction/VLFeedback"
    return train_dataset, eval_dataset


def get_mllm_dpo_dataloaders(mllm_dp_dataset, train_split, eval_split, processor, max_length, evaluate_llm, data_files, images_root):
    raw_train_dataset = load_dataset("json", data_files=data_files, split=train_split,trust_remote_code=True)
    raw_eval_dataset = load_dataset("json", data_files=data_files, split=eval_split,trust_remote_code=True)

    keep = ['image', 'question', 'rejected', 'chosen']
    remove = [key for key in raw_train_dataset.column_names if key not in keep]
    raw_train_dataset = raw_train_dataset.remove_columns(remove)
    raw_eval_dataset = raw_eval_dataset.remove_columns(remove)

    train_dataset = VQADataset(raw_train_dataset, mllm_dp_dataset, max_length, processor, images_root=images_root)
    eval_dataset = VQADataset(raw_eval_dataset, mllm_dp_dataset, max_length, processor, evaluate_llm=evaluate_llm, images_root=images_root)    
    return train_dataset, eval_dataset