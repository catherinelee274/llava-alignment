from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import TrainingArguments
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from customblip import ORPOBlipForVQA, ORPOConfig, PairBlipForVQA, DPOConfig, KTOConfig
import random, numpy as np, argparse
import torchmetrics.functional as FM
import torch.nn.functional as F
from torchmetrics.classification import MultilabelExactMatch
import evaluate as ev   #evaluate.list_evaluation_modules()
from langchain.evaluation.qa import QAEvalChain
from langchain_together import ChatTogether
import wandb
import time
import datetime
from peft import LoraConfig, get_peft_model

#custom data loaders:
from customdataloaders import get_vlfeedback_dataloaders, get_mllm_dpo_dataloaders

 
PROCESSOR_PRETRAINED_MODEL = "Salesforce/blip-vqa-base"
MAX_LENGTH = 64 #32 #max length for all text sequences: prompt, label, predicted label, negative label
TRAIN_SPLIT =   'train[:10]'  # 'train[:1%]'
EVAL_SPLIT = 'train[-10:]'
SPECIAL_TOKENS = [",",".","?","!", ":", ";", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "-", "_", "a", "an", "the"]
API_KEY = "b04b8ca599eb84354cde4bf4ab4cd32e13c832104c6668df71a6341e0478d50c" #Store somwhere else
LLM_QA_MODEL = "meta-llama/Llama-3-8b-chat-hf" #model used from huggingface hosted by together.ai 
VLFEEDBACKDATASET = "MMInstruction/VLFeedback"

LORA_TARGET_MODULES = ["query", "key", "value","dense",  "qkv", "fc1", "fc2", "projection"]

dataset_keys=['MMInstruction/VLFeedback', 'MLLM-DPO']
sft_methods = ["vanilla", "orpo", "dpo", "kto"]

def get_args():
    parser = argparse.ArgumentParser(description='Train a model with preference alignment')
    parser.add_argument('--alignment_method', choices=sft_methods, default=sft_methods[0], help='The alignment method to use (dpo, orpo, kto, or vanilla)')
    parser.add_argument('--pretrained_model', type=str, default="Salesforce/blip-vqa-base", help='Pretrained model from huggingface')
    parser.add_argument('--save_model', type=str, default="Model/blip-saved-model", help='Path to save the model')
    parser.add_argument("--push_to_hub", type=str, default="", help='Push the model to huggingface hub')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_scaler", type=bool, default=False)
    parser.add_argument("--train_percentage", type=int, default=1, help='if 0 then 10 records are used' )
    parser.add_argument("--eval_percentage", type=int, default=1, help='if 0 then 10 records are used')
    parser.add_argument("--max_eval_llm", type=int, default=0, help='maximum number of evaluation records for LLM per epoch')
    parser.add_argument("--iterations_loss", type=int, default=100, help='show training loss every x iterations')
    parser.add_argument("--wandb", type=bool, default=False, help='use wandb for logging')
    parser.add_argument("--wandb_project", type=str, default="cs231n_blip-vqa", help='wandb project name')
    parser.add_argument("--dataset", choices=dataset_keys, default=dataset_keys[0], help='which dataset to use')
    parser.add_argument("--mllm-dpo-json", help='path to MLLM-DPO dataset entries on disk')
    parser.add_argument("--mllm-dpo-images", help='path to images for MLLM-DPO dataset')    
    parser.add_argument("--orpo_lambda", type=float, default=0.5, help='lambda value for ORPO')
    parser.add_argument("--dpo_beta", type=float, default=0.5, help='beta value for DPO')
    parser.add_argument("--kto_beta", type=float, default=0.1, help='beta value for KTO')
    parser.add_argument("--lora", type=bool, default=False, help='use lora for peft') 
    parser.add_argument("--lora_rank", type=int, default=16, help='lora rank') 
    parser.add_argument("--lora_alpha", type=float, default=16, help='lora alpha')
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = torch.device("cpu")
    return device

def extendlabels(newlabels : torch.Tensor):
    #print(f"incoming newlabels.shape: {newlabels.shape}")
    if newlabels.shape[1] > MAX_LENGTH:
        newlabels.narrow(1,0,MAX_LENGTH)

    newlabels = F.pad(newlabels,(0,MAX_LENGTH-newlabels.size(1)),"constant",0)
    return newlabels

def nlp_f1_score(y_true : str, y_pred : str):
    for r in SPECIAL_TOKENS:
        y_true = y_true.replace(r, "")
        y_pred = y_pred.replace(r, "")
    true_words  = y_true.split(" ")
    pred_words = y_pred.split(" ")
 
    common_words = set(true_words).intersection(set(pred_words))
    precision = len(common_words) / len(pred_words)
    recall = len(common_words) / len(true_words)
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)
    #return f1_score(true_words, pred_words, average='weighted')

def evaluate_llm_score(eval_chain, questions, true_answers, predicted_answers):
    assert len(questions) == len(true_answers) == len(predicted_answers)
    examples = [ {'query':  q, 'answer': t }  for q,t in zip(questions, true_answers) ] 
    predicted_answers = [ {'result': p} for p in predicted_answers]
    #print(f"examples: {examples}")
    #print(f"predicted_answers: {predicted_answers}")
    graded_outputs = eval_chain.evaluate(examples, predicted_answers)
    total_correct = sum( (d['results'] == 'GRADE: CORRECT') for d in graded_outputs )
    return total_correct / len(questions)

def calculate_loss(batch, input_ids, pixel_values, attention_mask, labels, model, device, reference_model=None, alignment_method="vanilla") -> float:            
    if alignment_method == "vanilla":
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
    elif alignment_method == "orpo":
        neg_labels = batch['neg_labels']
        neg_labels = batch['neg_labels'].to(device)
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels,
                        neg_labels=neg_labels)
        loss = outputs.loss
    elif alignment_method == "dpo" or alignment_method == "kto":
        assert reference_model, "reference_model cannot be None for DPO"
        neg_labels = batch['neg_labels']
        neg_labels = batch['neg_labels'].to(device)        
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels,
                        neg_labels=neg_labels)
        with torch.no_grad():
            reference_outputs = reference_model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels,
                        neg_labels=neg_labels)
        if alignment_method == "kto":
            loss = model.calculate_kto_loss(outputs, reference_outputs, positive_labels=labels, negative_labels= neg_labels, attention_mask=attention_mask)
        else: #dpo 
            loss = model.calculate_dpo_loss(outputs, reference_outputs, positive_labels=labels, negative_labels= neg_labels, attention_mask=attention_mask)                    
                
    return loss

def main(args):
    alignment_method = args.alignment_method
         
    if args.dataset == dataset_keys[1]:
        assert args.mllm_dpo_json is not None, "--mllm-dpo-json must be specified if using MLLM-DPO dataset"
        assert args.mllm_dpo_images is not None, "--mllm-dpo-images must be specified if using MLLM-DPO dataset"
    num_epochs = args.epochs
    pretrained_model = args.pretrained_model  
    
    batch_size = args.batchsize
    learning_rate = args.lr
    iterations_show_loss = args.iterations_loss
    use_scaler = args.use_scaler
    vqa_dataset = args.dataset  #no longer hardcoded 

    if args.train_percentage == 0:
        train_split = TRAIN_SPLIT
    else:
        train_split = f'train[:{args.train_percentage}%]'
    if args.eval_percentage == 0:
        eval_split = EVAL_SPLIT
    else:
        eval_split = f'train[-{args.eval_percentage}%:]'
    max_eval_llm = args.max_eval_llm
    evaluate_llm = max_eval_llm > 0

    device = get_device()

    #device = torch.device("cpu")
    print("Using alignment method: ", alignment_method)
    if alignment_method == 'orpo':
        orpo_config = ORPOConfig(args.orpo_lambda)  
        print("Using ORPO with lambda: ", args.orpo_lambda)
    else:
        orpo_config = None
    if alignment_method == 'dpo':
        dpo_config = DPOConfig(args.dpo_beta)  
        print("Using DPO with beta: ", args.dpo_beta)
    else:
        dpo_config = None   
    if alignment_method == 'kto':
        kto_config = KTOConfig(args.kto_beta)  
        print("Using KTO with beta: ", args.kto_beta)
    print("Dataset: ", vqa_dataset)
    print("Pretrained model: ", pretrained_model)
    print("Save model to: ", args.save_model)
    if args.push_to_hub != "":
        print("Push model to hub: ", args.push_to_hub)
    print("Using dataset: ", vqa_dataset)
    print(f"Using device: {device}")
    print("Use scaler: ", use_scaler)
    print("Number of epochs: ", num_epochs)
    print("Train split: ", train_split)
    print("Eval split: ", eval_split)
    print("Batch size: ", batch_size)
    print("Learning rate: ", learning_rate)
    print("Iterations show loss every: ", iterations_show_loss, " iterations")
    if max_eval_llm > 0:
        print("Maximum number of evaluation records for LLM per epoch: ", max_eval_llm)
    else:
        print("LLM not used for evaluation")

    if args.wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)
        wandb.config.update({"alignment_method": alignment_method})
        if alignment_method == "orpo":
            wandb.config.update({"orpo_lambda": args.orpo_lambda})
        if alignment_method == "dpo":
            wandb.config.update({"dpo_beta": args.dpo_beta}) 
        if alignment_method == "kto":
            wandb.config.update({"kto_beta": args.kto_beta})           
        wandb.config.update({"device": str(device)})
        wandb.config.update({"dataset":  vqa_dataset})
        wandb.config.update({"pretrained_model": pretrained_model}) 
        wandb.config.update({"push_to_hub": args.push_to_hub})
        wandb.config.update({"save_model": args.save_model})
        wandb.config.update({"dataset": vqa_dataset})  
        wandb.config.update({"train_split": train_split})
        wandb.config.update({"eval_split": eval_split})
        wandb.config.update({"batch_size": batch_size})
        wandb.config.update({"learning_rate": learning_rate})
        wandb.config.update({"num_epochs": num_epochs})
        wandb.config.update({"iterations_show_loss": iterations_show_loss})
        wandb.config.update({"max_eval_llm": max_eval_llm})
        wandb.config.update({"evaluate_llm": evaluate_llm})
        wandb.config.update({"use_scaler": use_scaler})
        wandb.config.update({"lora": args.lora})
        wandb.config.update({"lora_rank": args.lora_rank})
        wandb.config.update({"lora_alpha": args.lora_alpha})


    processor  = BlipProcessor.from_pretrained(PROCESSOR_PRETRAINED_MODEL) 
    
    reference_model = None
    if alignment_method == "vanilla":
        model = BlipForQuestionAnswering.from_pretrained(pretrained_model).to(device)   
    elif alignment_method == "orpo":
        model = ORPOBlipForVQA.from_pretrained(pretrained_model,orpo_config=orpo_config).to(device) 
    elif alignment_method == "dpo":
        model = PairBlipForVQA.from_pretrained(pretrained_model, dpo_config=dpo_config).to(device)
    elif alignment_method == "kto":
        model = PairBlipForVQA.from_pretrained(pretrained_model, kto_config=kto_config).to(device)

    if alignment_method == "dpo" or alignment_method == "kto":
        reference_model = PairBlipForVQA.from_pretrained(pretrained_model).to(device) #dpo/kto_config not used in reference model; could as well use sft BlipForQuestionAnswering if were not for logps results
        # Freeze all layers in reference model
        for param in reference_model.parameters():
            param.requires_grad = False

    #for name, p in model.named_parameters():
    #    if p.requires_grad:
    #        print(name)
    if args.lora:
        lora_config = LoraConfig( r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", target_modules=LORA_TARGET_MODULES )
        model = get_peft_model(model, lora_config)
        print("Using Lora for PEFT")
        model.print_trainable_parameters()
    else:
        print("Not using Lora for PEFT")

    eval_chain = QAEvalChain.from_llm( llm= ChatTogether(  together_api_key=API_KEY, model=LLM_QA_MODEL, ))

    #   Make sure that dataset is converted into the following format:
    #   (later add rejected answers)
    #   batched_data = {
    #           'input_ids':  input_ids,
    #           'pixel_values':   pixel_values,
    #           'attention_mask':  attention_mask,
    #           'labels': labels,
    #           'neg_labels': neg_labels,   #create from rejected
    #           'questions': questions,  
    #           'true_answers': true_answers
    #       }
    #   train_dataset and eval_dataset need .collate_fn() method

    if vqa_dataset == dataset_keys[0]: 
        train_dataset, eval_dataset = get_vlfeedback_dataloaders(vqa_dataset, train_split, eval_split, processor, MAX_LENGTH, evaluate_llm)
    elif vqa_dataset == dataset_keys[1]: 
        train_dataset, eval_dataset = get_mllm_dpo_dataloaders(vqa_dataset, train_split, eval_split, processor, MAX_LENGTH, evaluate_llm, data_files=args.mllm_dpo_json, images_root=args.mllm_dpo_images)
    else: 
        print(f"Dataset {vqa_dataset} not supported")
        exit()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=eval_dataset.collate_fn)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
 
    min_eval_loss = float("inf")

    best_model = {
        "epoch": 0,
        "loss": float("inf"),
        "EM": 0,
        "F1": 0,
        "LLM Eval:": 0
    }
 
    tracking_information = []
    iterations = 0

    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()

    print("Starting the finetuning process!")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        epoch_loss = 0
        epoch_iterations = 0
        epoch_start_time = time.time()
        iterations_start_time = time.time()
        model.train()
        for batch in train_dataloader:               #issues with tqdm
            optimizer.zero_grad()
   
            input_ids =  batch['input_ids'].to(device) 
            pixel_values =  batch['pixel_values'].to(device) 
            attention_mask =  batch['attention_mask'].to(device) 
            labels =  batch['labels'].to(device)  
            loss = calculate_loss(batch, input_ids, pixel_values, attention_mask, labels, model, device, reference_model=reference_model, alignment_method=alignment_method)        
            l = loss.item()

            if not np.isnan(l):      #don't add NaN values to the loss. If loss was NaN before then this way we know if at some point if becomes less than NaN
                epoch_loss += l 
            elif epoch_loss == 0:   
                epoch_loss = l  #initial loss value. Assign NaN if necessary;
            
            last_loss_had_nan = np.isnan(l)

            if not use_scaler:
                loss.backward()
                optimizer.step() 
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            iterations = iterations + 1
            epoch_iterations += 1
            if iterations % iterations_show_loss == 0:
                iterations_time = time.time() - iterations_start_time
                iterations_start_time = time.time()
                print(f"Iteration {iterations} - Epoch {epoch+1} Training Loss: {epoch_loss/epoch_iterations } - Iterations per second: {iterations_show_loss/iterations_time :.2f} - Examples per second: {iterations_show_loss/iterations_time*batch_size :.2f}")
                if args.wandb:
                    wandb.log({ 'Iteration': iterations, 'Epoch': epoch+1, 'Training Loss': epoch_loss/epoch_iterations, 'Iterations per second': round(iterations_show_loss/iterations_time,2), 'Examples per second': round(iterations_show_loss/iterations_time*batch_size,2)})
                if last_loss_had_nan:
                    print("Last loss was NaN")
        
        epoch_end_time = eval_start_time = time.time()   
        output_epoch_time = str(datetime.timedelta(seconds=round(epoch_end_time - epoch_start_time)))
        print(f"Epoch {epoch+1} Training Time: {output_epoch_time}") 
        if args.wandb:
            wandb.log({ 'Epoch': epoch+1, 'Training Time': output_epoch_time})    
        #Start evaluation:  
        optimizer.zero_grad() 
        model.eval()
        eval_loss = 0
        print("Evaluating the model for epoch: ", epoch+1)
        total = 0
        total_correct = 0
        total_f1 = 0
        llm_questions = []
        llm_true_answers = []
        llm_predicted_answers = []

        for batch in  valid_dataloader:


            input_ids =  batch['input_ids'].to(device) 
            pixel_values =  batch['pixel_values'].to(device) 
            attention_mask =  batch['attention_mask'].to(device) 
            labels =  batch['labels'].to(device) 
            questions = batch['questions']
            true_answers = batch['true_answers'] 

            loss = calculate_loss(batch, input_ids, pixel_values, attention_mask, labels, model, device, reference_model=reference_model, alignment_method=alignment_method) 
            l = loss.item()

            eval_loss += l 

            predict_out = model.generate(input_ids=input_ids,
                            pixel_values=pixel_values, max_length=MAX_LENGTH)
            
            predicted_answers = processor.batch_decode(predict_out, skip_special_tokens=True)

            for t,p in zip(true_answers,predicted_answers):
                f1_b = nlp_f1_score(t,p)
                total_f1 += f1_b
                if t == p:
                    total_correct += 1
                total += 1
            if evaluate_llm and len(llm_questions) < max_eval_llm:
                llm_questions.extend(questions)
                llm_true_answers.extend(true_answers)
                llm_predicted_answers.extend(predicted_answers)
           
        eval_loss = eval_loss/len(valid_dataloader)
        epoch_loss = epoch_loss/len(train_dataloader)
        
        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss, eval_loss, optimizer.param_groups[0]["lr"]))

        em = round(total_correct/total,3)
        f1 = round(total_f1/total,3)
 
        print(f"Exact Match: {em},  F1: {f1}") 

        if evaluate_llm:
            llm_eval_score = evaluate_llm_score(eval_chain, llm_questions, llm_true_answers, llm_predicted_answers) 
            print(f"LLM Evaluation Score: {llm_eval_score}")
            if args.wandb:
                wandb.log({ 'LLM Evaluation Score': llm_eval_score})
        else:
            llm_eval_score = 0
        if args.wandb:
            wandb.log({ 'Epoch': epoch+1, 'Training Loss': epoch_loss, 'Eval Loss': eval_loss, 'Learning Rate': optimizer.param_groups[0]["lr"]})
            wandb.log({ 'Exact Match': em, 'F1': f1})
        if eval_loss < min_eval_loss:
            model.save_pretrained(args.save_model, from_pt=True) 
            print(f"Saved model to {args.save_model}")
            min_eval_loss = eval_loss
            best_model["epoch"] = epoch+1
            best_model["loss"] = eval_loss
            best_model["EM"] = em
            best_model["F1"] = f1
            if evaluate_llm:
                best_model["LLM Eval:"] = llm_eval_score
        else:
            print("Model not saved. Eval loss is higher than the minimum eval loss")

        tracking_information.append((epoch, epoch_loss, eval_loss, em, f1, llm_eval_score, optimizer.param_groups[0]["lr"]))    
        scheduler.step()
        print("Learning rate: ", optimizer.param_groups[0]["lr"])
        eval_end_time = time.time()  
        output_eval_epoch_time = str(datetime.timedelta(seconds=round(eval_end_time - eval_start_time)))
        print(f"Epoch {epoch+1} Evaluation Time: {output_eval_epoch_time}")   
        if args.wandb:
            wandb.log({ 'Epoch': epoch+1, 'Evaluation Time': output_eval_epoch_time})  
        
        
    print("The finetuning process is done!")
    print("Best model: ", best_model)
    if args.wandb:
        wandb.log(best_model)
        wandb.finish()
    if args.push_to_hub != "":
        print(f"Push model to hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        print("Pushed to hub complete")


if __name__ == "__main__":
    args = get_args()
    main(args)
