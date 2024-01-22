import os
import json
import sys
from openai import OpenAI
from math import exp
import numpy as np
from dotenv import load_dotenv
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY") 
vectordb_keys = os.getenv("OPENAI_MODEL") 
# print("Here:>>>" + str(openai_api_key))
# os.environ["OPENAI_API_KEY"] = openai_api_key
# os.environ["VECTORDB_MODEL"] = vectordb_keys
client = OpenAI(api_key=openai_api_key)


def get_completion(
    messages: list[dict[str, str]],
    model: str = vectordb_keys,
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """Return the completion of the prompt.
    @parameter messages: list of dictionaries with keys 'role' and 'content'.
    @parameter model: the model to use for completion. Defaults to 'davinci'.
    @parameter max_tokens: max tokens to use for each prompt completion.
    @parameter temperature: the higher the temperature, the crazier the text
    @parameter stop: token at which text generation is stopped
    @parameter seed: random seed for text generation
    @parameter tools: list of tools to use for post-processing the output.
    @parameter logprobs: whether to return log probabilities of the output tokens or not.
    @returns completion: the completion of the prompt.
    """

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def file_reader(path: str, ) -> str:
    fname = os.path.join(path)
    with open(fname, 'r') as f:
        system_message = f.read()
    return system_message
            

def generate_test_data(prompt: str, context: str, num_test_output: str) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    API_RESPONSE = get_completion(
        [
            {
                "role": "user", 
                "content": prompt.replace("{context}", context).replace("{num_test_output}", num_test_output)
            }
        ],
        model=vectordb_keys,
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = API_RESPONSE.choices[0].message.content
    return system_msg


def main(num: str):
    context_message = file_reader("../prompts/context.txt")
    prompt_message = file_reader("../prompts/prompt-generating-prompt.txt")
    context = str(context_message)
    prompt = str(prompt_message)

    generate_prompts = generate_test_data(prompt, context, num)

    def save_txt(generate_prompts) -> None:
        # Specify the file path
        file_path = "../prompts/automatically-generated-prompts.txt"
        with open(file_path, 'w') as txt_file:
            txt_file.write(generate_prompts)
        
        print(f"Generated Prompts have been saved to {file_path}")

    save_txt(generate_prompts)

    print("===========")
    print("Prompts")
    print("===========")
    print(generate_prompts)


if __name__ == "__main__":
    main("5") # n number of prompts to generate