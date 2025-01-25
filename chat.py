import requests
import json
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import urllib3
import time
from requests.exceptions import Timeout, RequestException

def load_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load and validate the configuration from the specified Python file.
    
    This function loads model configurations and ensures all required fields are present.
    It handles both absolute and relative paths, converting them to proper Path objects.
    """
    try:
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
            
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Validate configurations
        configs = {"llama": config.LLAMA_CONFIG, "qwen": config.QWEN_CONFIG}
        for model_name, config_dict in configs.items():
            if not isinstance(config_dict, dict):
                raise ValueError(f"Invalid configuration format for {model_name}")
            required_keys = ["api_url", "api_key", "model"]
            missing_keys = [key for key in required_keys if key not in config_dict]
            if missing_keys:
                raise ValueError(f"Missing required keys for {model_name}: {missing_keys}")
                
        return configs
        
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {str(e)}")

def parse_llm_response(response_text: str) -> str:
    """
    Parse the API response, handling both string and potential JSON responses.
    
    The function attempts to decode JSON if the response looks like JSON,
    otherwise returns the string directly after cleaning it.
    """
    try:
        # If the response is a JSON string, parse it
        if response_text.strip().startswith('{'):
            response_json = json.loads(response_text)
            # Handle various possible JSON structures
            if isinstance(response_json, dict):
                if "response" in response_json:
                    return response_json["response"]
                elif "generated_text" in response_json:
                    return response_json["generated_text"]
                elif "choices" in response_json and isinstance(response_json["choices"], list):
                    return response_json["choices"][0]["text"]
            return str(response_json)
        
        # If it's a plain string (possibly quoted), clean it up
        cleaned_response = response_text.strip('"\'')
        return cleaned_response
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return the cleaned string
        return response_text.strip('"\'')
    except Exception as e:
        print(f"Warning: Error while parsing response: {str(e)}")
        return response_text

def chat_with_llm(prompt: str, model_config: Dict[str, str], 
                 max_retries: int = 3, initial_timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Send a chat message to the LLM API and handle the response.
    
    This function manages the complete interaction with the API, including:
    - Input validation
    - Request preparation and sending
    - Response handling and parsing
    - Error management and retries
    """
    if not prompt.strip():
        return None, "Empty prompt provided"
    
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": model_config["api_key"]
    }
    
    data = {
        "temperature": 0.7,
        "prompt": prompt,
        "repeat_penalty": 1.0
    }
    
    # Print request details for debugging
    print(f"\nDebug - API URL: {model_config['api_url']}")
    print(f"Debug - Request Data: {json.dumps(data, indent=2)}")
    
    timeout = initial_timeout
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                model_config["api_url"],
                headers=headers,
                json=data,
                timeout=timeout,
                verify=False
            )
            
            print(f"Debug - Response Status Code: {response.status_code}")
            print(f"Debug - Raw Response: {response.text}")
            
            if response.status_code == 200:
                parsed_response = parse_llm_response(response.text)
                if parsed_response:
                    return parsed_response, None
                return None, "Empty response from API"
            else:
                error_msg = f"API returned status {response.status_code}"
                if response.text:
                    error_msg += f": {response.text}"
                return None, error_msg
                
        except Timeout:
            timeout *= 2
            if attempt < max_retries - 1:
                print(f"Request timed out, retrying with timeout={timeout}s...")
                time.sleep(1)
                continue
            return None, f"API request timed out after {max_retries} attempts"
            
        except RequestException as e:
            return None, f"API request failed: {str(e)}"
            
    return None, "Max retries exceeded"

def main():
    """
    Main function that handles the interactive chat loop.
    
    This function manages:
    - Configuration loading
    - Model selection
    - User input processing
    - Response display
    - Error handling
    """
    config_path = r"M:\ML\signatejpx\secret\config.py"
    try:
        configs = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    print("Available models:")
    print("1. Llama (70B)")
    print("2. Qwen (72B)")
    
    while True:
        model_choice = input("Select model (1 or 2): ").strip()
        if model_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    model_config = configs["llama"] if model_choice == "1" else configs["qwen"]
    print(f"\nUsing {model_config['model']} model")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                print("Please enter a non-empty prompt")
                continue
                
            response, error = chat_with_llm(user_input, model_config)
            if response:
                print(f"\nLLM: {response}")
            elif error:
                print(f"\nError: {error}")
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()