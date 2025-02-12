from transformers import BlipForQuestionAnswering
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class ORPOConfig:
    def __init__(self, orpo_lambda):
        self.orpo_lambda = orpo_lambda

@dataclass
class ORPOBlipTextVisionModelOutput():
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ORPOBlipForVQA(BlipForQuestionAnswering):
    def __init__(self, config, orpo_config=None):
        super().__init__(config)
        self.orpo_config = orpo_config

    def _calculate_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, attention_mask: torch.LongTensor=None):
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)

    def calculate_orpo_loss(self, outputs_pos, outputs_neg, positive_labels, negative_labels, attention_mask):
        #check self.tokenizer.pad_token_id

        orpo_lambda=self.orpo_config.orpo_lambda

        pos_loss = outputs_pos.loss

        pos_prob = self._calculate_logps(outputs_pos.logits, positive_labels, attention_mask)

        neg_prob = self._calculate_logps(outputs_neg.logits, negative_labels, attention_mask)
                              
        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
            
        # Calculate the Final Loss
        loss = torch.mean(pos_loss - orpo_lambda * ratio)      
        return loss
   
    def forward( 
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            neg_labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None   
        ):

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict 
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels, #positive labels
            return_dict=return_dict,
            reduction="mean",
        )

        if labels is not None:
            if self.orpo_config == None:
                decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
            else:           
                neg_answer_output = self.text_decoder(
                                        input_ids=decoder_input_ids,
                                        attention_mask=decoder_attention_mask,
                                        encoder_hidden_states=question_embeds,
                                        encoder_attention_mask=attention_mask,
                                        labels=neg_labels,
                                        return_dict=return_dict,
                                        reduction="mean"
                            )
                decoder_loss = self.calculate_orpo_loss(outputs_pos=answer_output, 
                                                        outputs_neg=neg_answer_output, 
                                                        positive_labels = labels, 
                                                        negative_labels = neg_labels,  
                                                        attention_mask=attention_mask
                                                        )
        else:
            decoder_loss = None

        return ORPOBlipTextVisionModelOutput(
            loss=decoder_loss,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

class DPOConfig:
    def __init__(self, dpo_beta):
        self.beta = dpo_beta

class KTOConfig:
    """
    KTO paper https://arxiv.org/abs/2402.01306 , code loosely referred to https://github.com/ContextualAI/HALOs
    """
    def __init__(self, kto_beta, lambda_D=1, lambda_U=1):
        #lambda_D and lambda_U used if imbalance between positive and negative samples
        #not used here. Lambda = 1 for all
        self.beta = kto_beta
        self.lamda = 1

@dataclass
class PairBlipTextVisionModelOutput():
    loss: Optional[torch.FloatTensor] = None
    positive_logits: Optional[torch.FloatTensor] = None
    negative_logits: Optional[torch.FloatTensor] = None


class PairBlipForVQA(BlipForQuestionAnswering):
    """
    PairBlipForVQA is a class that extends BlipForQuestionAnswering 
    to calculate either the KTO or DPO loss for the VQA task.
    This assumes that each batch has a positive and negative sample.
    Different upair class needs to be implemented to handle batch with single answer
    and indicator whether positive or negative
    """
    def __init__(self, config, dpo_config=None, kto_config=None):
        super().__init__(config)
        self.dpo_config = dpo_config
        self.kto_config = kto_config

    def _calculate_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, attention_mask: torch.LongTensor=None):
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * loss_mask).sum(-1)
    
    def _r(self, logits, labels, attention_mask):
        logps = self._calculate_logps(logits, labels, attention_mask)
        return torch.exp(logps)

    def forward( 
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            neg_labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            dpo_config: Optional[DPOConfig] = None   
        ):

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict 
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )

        if labels is not None and decoder_input_ids is None: 
            decoder_input_ids = labels

        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels, #positive labels
            return_dict=return_dict,
            reduction="mean",
        )

        if labels is not None:
            decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
           
            neg_answer_output = self.text_decoder(
                                        input_ids=decoder_input_ids,
                                        attention_mask=decoder_attention_mask,
                                        encoder_hidden_states=question_embeds,
                                        encoder_attention_mask=attention_mask,
                                        labels=neg_labels,
                                        return_dict=return_dict,                                 
                                        reduction="mean"
                            )
        else:
            decoder_loss = None

        #Return log probabilities for chosen and rejected
        return PairBlipTextVisionModelOutput(
            loss=decoder_loss,  #loss is from positive anwer
            positive_logits=answer_output.logits,
            negative_logits=neg_answer_output.logits,
        )
    
    def calculate_dpo_loss(self, policy_outputs, reference_outputs, positive_labels, negative_labels, attention_mask):

        policy_chosen_logps = self._calculate_logps(policy_outputs.positive_logits, positive_labels, attention_mask)

        policy_rejected_logps = self._calculate_logps(policy_outputs.negative_logits, negative_labels, attention_mask)

        reference_chosen_logps = self._calculate_logps(reference_outputs.positive_logits, positive_labels, attention_mask)

        reference_rejected_logps = self._calculate_logps(reference_outputs.negative_logits, negative_labels, attention_mask)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.dpo_config.beta * logits)

        loss = losses.mean()    
    
        return loss


    def _r_t(self, policy_logps, reference_logs):
        return policy_logps - reference_logs
    
    
    def _z0(self, pol_chosen_logps, pol_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        #Calculate KL Estimate z_hat_0 See https://arxiv.org/pdf/2402.01306 page 6 
        m = pol_chosen_logps - pol_rejected_logps - ref_chosen_logps + ref_rejected_logps
        return  max(0, m.mean())

    def calculate_kto_loss(self, policy_outputs, reference_outputs, positive_labels, negative_labels, attention_mask):
        #See https://arxiv.org/pdf/2402.01306 page 5 for math for LKTO

        pol_chosen_logps = self._calculate_logps(policy_outputs.positive_logits, positive_labels, attention_mask)

        pol_rejected_logps = self._calculate_logps(policy_outputs.negative_logits, negative_labels, attention_mask)

        ref_chosen_logps = self._calculate_logps(reference_outputs.positive_logits, positive_labels, attention_mask)

        ref_rejected_logps = self._calculate_logps(reference_outputs.negative_logits, negative_labels, attention_mask)

        #Per paper: z0 is the estimated KL divergence between the two distributions. Also, as in paper, for stability reasons do not
        #backpropagate through z0. Hence torch.no_grad()
               
        with torch.no_grad():
            z0 = self._z0(pol_chosen_logps, pol_rejected_logps, ref_chosen_logps, ref_rejected_logps)

        #Both Lambda_D and Lambda_U are set to 1
        losses = 2 - F.sigmoid(self.kto_config.beta * ( self._r_t(pol_chosen_logps, ref_chosen_logps) - z0 ))  \
                   - F.sigmoid(self.kto_config.beta * ( z0 - self._r_t(pol_rejected_logps, ref_rejected_logps)) )
        loss = losses.mean()    
    
        return loss
    

 