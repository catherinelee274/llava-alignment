# llava_alignment


ther model finetune with dpo, kto, etc
llava w dpo

llava kto

llava KTO > DPO ~= CPO > IPO

ORPO

"Several techniques for preference alignment mitigate the need for reinforcement learning (Rafailov et al., 2023; Song et al., 2023; Azar et al., 2023; Ethayarajh et al., 2024). Rafailov et al. (2023) introduce direct policy optimization (DPO), which combines the reward modeling stage into the preference learning stage. Azar et al. (2023) prevented potential overfitting problems in DPO through identity preference optimization (IPO). Ethayarajh et al. (2024) and Cai et al. (2023) proposed Kahneman-Tversky Optimisation (KTO) and Unified Language Model Alignment (ULMA) that does not require the pair-wise preference dataset, unlike RLHF and DPO. Song et al. (2023) further suggests incorporation of the softmax value of the reference response set in the negative log-likelihood loss to merge the supervised fine-tuning and preference alignment. "

https://github.com/Unified-Language-Model-Alignment/src

https://huggingface.co/alexshengzhili/llava-v1.5-13b-dpo

https://github.com/findalexli/mllm-dpo/blob/3059f3e77716fe12289b98bbf1676ff6061e8bf7/llava/train/train.py#L854
