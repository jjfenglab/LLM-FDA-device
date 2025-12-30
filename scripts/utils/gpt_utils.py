from typing import Dict, Tuple
import json

from openai import OpenAI

def generate_a_string_from_gpt(instruction: str, client: OpenAI, seed: int, model='gpt-4o') -> str:
    """Call GPT to generate a string with a complete instruction.
    
    Args:
        instruction (str): The instruction to send to the model
        client (OpenAI): OpenAI client instance
        seed (int): Random seed for reproducibility
        model (str): Model name to use, defaults to 'gpt-4o'
        
    Returns:
        str: Generated text from the model
    """
    # Call GPT to get all device information
    response = client.chat.completions.create(
      model=model,
      seed=seed,
      messages=[
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": instruction},
      ]
    )
    return response.choices[0].message.content 

def calculate_gpt_cost(
    model: str,
    usage: dict,
) -> float:
    """
    Calculate the cost in USD for a GPT API call based on usage information.
    
    Args:
        model: The model name
        usage: Dictionary containing token usage details from OpenAI API response
            Example:
            {
                "prompt_tokens": 2006,
                "completion_tokens": 300,
                "total_tokens": 2306,
                "prompt_tokens_details": {
                    "cached_tokens": 1920
                }
            }
    """
    # Base pricing (regular_input_price, cached_input_price, output_price) per 1M tokens
    pricing = {
        'gpt-4o': (2.50, 1.25, 10.00),  # $2.50/$1.25/$10.00 per 1M tokens
        'gpt-4o-2024-11-20': (2.50, 1.25, 10.00),  # $2.50/$1.25/$10.00 per 1M tokens
        'gpt-4o-2024-08-06': (2.50, 1.25, 10.00),  # $2.50/$1.25/$10.00 per 1M tokens
        'gpt-4o-mini': (0.15, 0.075, 0.60),  # $0.15/$0.075/$0.60 per 1M tokens
        'gpt-4o-mini-2024-07-18': (0.15, 0.075, 0.60),  # $0.15/$0.075/$0.60 per 1M tokens
        'gpt-4.1': (2.00, 0.50, 8.00),        # Input $2.00, Cached $0.50, Output $8.00 / 1M tokens
        'gpt-4.1-mini': (0.40, 0.10, 1.60),   # Input $0.40, Cached $0.10, Output $1.60 / 1M tokens
        'gpt-4.1-nano': (0.10, 0.025, 0.40),  # Input $0.10, Cached $0.025, Output $0.40 / 1M tokens
    }
    
    if model not in pricing:
        return
        # raise ValueError(f"Unknown model: {model}")
    
    regular_input_price, cached_input_price, output_price = pricing[model]
    
    # Extract token counts from usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)
    
    # Calculate regular and cached input costs separately
    regular_tokens = max(0, input_tokens - cached_tokens)
    regular_input_cost = (regular_tokens / 1_000_000) * regular_input_price
    cached_input_cost = (cached_tokens / 1_000_000) * cached_input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    
    return regular_input_cost + cached_input_cost + output_cost

def get_device_info_with_cost(
    client: OpenAI,
    instruction: str,
    model: str, 
    seed: int,
) -> Tuple[Dict, float]:    
    """Use GPT to extract info about a device and return the cost of the API call.
    
    Returns:
        Tuple containing:
        - device_info: the parsed JSON response
        - cost: the cost of the API call in USD
    """
    messages = [
        {"role": "system", "content": 'You are a helpful assistant who always responds with valid JSON.'},
        {"role": "user", "content": instruction},
    ]
    
    response = client.chat.completions.create(
      model=model,
      seed=seed,
      response_format={ "type": "json_object" },
      messages=messages
    )
    
    # Calculate cost from usage
    cost = calculate_gpt_cost(
        model=model,
        usage=response.usage
    )
    
    try:
        return json.loads(response.choices[0].message.content), cost
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response content: {response.choices[0].message.content}")
        # Return empty dict as fallback
        return {}, cost