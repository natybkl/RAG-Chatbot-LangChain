import os
import json
import sys
from openai import OpenAI
from math import exp
import numpy as np
from utility.env_manager import get_env_manager
from scripts import prompt_generator


env_manager = get_env_manager()
client = OpenAI(api_key=env_manager['openai_keys']['OPENAI_API_KEY'])


def evaluate(prompt: str, user_message: str, context: str, use_test_data: bool = False) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    num_test_output = str(10)
    API_RESPONSE = prompt_generator.get_completion(
        [
            {
                "role": "system", 
                "content": prompt.replace("{Context}", context).replace("{Question}", user_message)
            }
        ],
        model=env_manager['vectordb_keys']['VECTORDB_MODEL'],
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = str(API_RESPONSE.choices[0].message.content)

    for i, logprob in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
        output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100,2)}%\n'
        print(output)
        
        if system_msg == 'true' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'true'
        elif system_msg == 'false' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'false'
        else:
            classification = 'false'

    return classification

if __name__ == "__main__":
    context_message = prompt_generator.file_reader("prompts/context.txt")
    prompt_message = prompt_generator.file_reader("prompts/generic-evaluation-prompt.txt")
    context = str(context_message)
    prompt = str(prompt_message)
    
    user_message = str(input("question: "))
    
    print(evaluate(prompt, user_message, context))