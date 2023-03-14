import os
import io
import logging
import slack_sdk
from slack_bolt import App
from slack_bolt.response import BoltResponse
from slack_bolt.error import BoltUnhandledRequestError
from slack_bolt.adapter.socket_mode import SocketModeHandler
from torch import autocast
from slack_sdk.errors import SlackApiError


import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import pegasuslm

logging.basicConfig(level=logging.INFO)

BOT_CHANNEL = "C04TLHRGYE6"

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    raise_error_for_unhandled_request=True,
)

model_id = "range3/pegasus-gpt2-medium"
device = "cuda"

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)
model.half()


@app.event(
    event={
        "type": "message",
        "subtype": None,
    },
    matchers=[lambda body: body["event"]["channel"] == BOT_CHANNEL],
)
def handle_message(
    ack, logger: logging.Logger, event, client: slack_sdk.web.client.WebClient
):
    ack()
    prompt = event["text"]
    logger.info(prompt)
    text = generate_text(prompt)
    try:
        # Call the chat.postMessage method using the WebClient
        result = client.chat_postMessage(
            channel=BOT_CHANNEL,
            text=text,
        )
        logger.info(result)

    except SlackApiError as e:
        logger.error(f"Error posting message: {e}")


@app.error
def handle_errors(error):
    if isinstance(error, BoltUnhandledRequestError):
        return BoltResponse(status=200, body="")
    else:
        return BoltResponse(status=500, body="Something Wrong")


def main():
    SocketModeHandler(
        app=app,
        app_token=os.environ["SLACK_APP_TOKEN"],
        trace_enabled=True,
    ).start()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_text(prompt: str):
    set_seed(42)

    stop_token = "</s>"

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
        max_length=300,
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
    text = text[: text.find(stop_token)].strip()

    return text


if __name__ == "__main__":
    main()
