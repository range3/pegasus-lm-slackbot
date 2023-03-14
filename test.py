# import argparse
import logging

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from pegasuslm import (
    PegasusGPT2Config,
    PegasusGPT2Tokenizer,
    PegasusGPT2Model,
)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    # Initialize the model and tokenizer
    model_id = "range3/pegasus-gpt2-medium"
    device = "cuda"
    length = 1024
    stop_token="</s>"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.half()

    prompt = "吾輩は猫である。"

    encoded_prompt = tokenizer(
        prompt,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    input_len = len(encoded_prompt["input_ids"][0])
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        **encoded_prompt,
        max_length=1024,
        temperature=0.8,
        top_k=500,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        num_return_sequences=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        pad_token_id=tokenizer.eos_token_id,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequence = output_sequences[0].tolist()[input_len:]
    text = tokenizer.decode(
        generated_sequence,
        clean_up_tokenization_spaces=True,
        # skip_special_tokens=True,
    )

    # Remove all text after the stop token
    text = text[:text.find(stop_token)].strip()

    print(text)

if __name__ == "__main__":
    main()
