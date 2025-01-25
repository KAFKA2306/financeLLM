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
    """
    try:
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
            
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        configs = {"llama": config.LLAMA_CONFIG, "qwen": config.QWEN_CONFIG}
        # Validate configurations
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

def parse_llm_response(response_json: Dict, model_type: str) -> Optional[str]:
    """
    Parse the API response based on the model type.
    Handles different response formats for Llama and Qwen models.
    """
    try:
        # Print response for debugging
        print(f"\nDebug - Raw API Response: {json.dumps(response_json, indent=2)}")
        
        if model_type == "llama":
            # Llama model response format
            if "response" in response_json:
                return response_json["response"]
            elif "generated_text" in response_json:
                return response_json["generated_text"]
            elif "output" in response_json:
                return response_json["output"]
            else:
                print(f"Warning: Unexpected Llama response format. Available keys: {list(response_json.keys())}")
                return str(response_json)
        else:
            # Qwen model response format
            if "choices" in response_json and isinstance(response_json["choices"], list):
                return response_json["choices"][0]["text"]
            else:
                print(f"Warning: Unexpected Qwen response format. Available keys: {list(response_json.keys())}")
                return str(response_json)
                
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return None

def chat_with_llm(prompt: str, model_config: Dict[str, str], 
                 max_retries: int = 3, initial_timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Send a chat message to the LLM API with improved response handling and debugging.
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
    model_type = "llama" if "llama" in model_config["model"].lower() else "qwen"
    
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
            
            if response.status_code == 200:
                response_json = response.json()
                parsed_response = parse_llm_response(response_json, model_type)
                if parsed_response:
                    return parsed_response, None
                return None, "Failed to parse API response"
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
            
        except json.JSONDecodeError as e:
            return None, f"Failed to decode API response: {str(e)}"
            
    return None, "Max retries exceeded"

def main():
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