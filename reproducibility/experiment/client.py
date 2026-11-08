#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenRouter chat-completions client shared by the API experiments.

The API key is read from OPENROUTER_API_KEY. Every call is stateless.
"""

import os
import uuid
import time
import requests
import threading
from typing import List, Dict, Optional, Any

class Client:
    """Thread-safe OpenRouter client; each instance is independent."""
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1", request_user_tag: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        self.base_url = base_url
        self.request_user_tag = request_user_tag
        # No shared lock - each Client instance is independent

    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = 0.0,
             max_tokens: int = 600,
             top_p: float = 1.0,
             frequency_penalty: float = 0.0,
             presence_penalty: float = 0.0,
             stop: Optional[List[str]] = None,
             run_id: Optional[str] = None,
             retries: int = 2,
             timeout: int = 120) -> Dict[str, Any]:
        """
        Single stateless call with enhanced error handling - Thread Safe
        Returns dict with 'content', 'finish_reason', 'usage', and 'model' fields
        """
        # Special handling for DeepSeek R1: top_p=1.0 may cause empty responses
        if 'deepseek' in model.lower() and top_p >= 0.99:
            top_p = 0.95  # Cap at 0.95 for DeepSeek
            
        # Special handling for O1: doesn't support temperature/top_p parameters
        if 'o1' in model.lower():
            temperature = 1.0  # O1 ignores temperature
            top_p = 1.0  # O1 ignores top_p
        
        # Each Client instance is independent - no shared state
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
        }
        
        # Add stop sequences if provided
        if stop is not None:
            payload["stop"] = stop
        
        # Note: Most commercial models (GPT, Claude, Gemini) don't support seed parameter
        # We skip seed setting to avoid API errors

        if self.request_user_tag:
            payload["user"] = str(self.request_user_tag)
        if run_id:
            payload["metadata"] = {"run_id": run_id}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://local.experiment/persuasion",
            "X-Title": f"LLM Persuasion Experiment {run_id or uuid.uuid4()}",
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                
                # Extract response information
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "")
                usage = data.get("usage", {})
                model_name = data.get("model", model)
                
                return {
                    "content": content,
                    "finish_reason": finish_reason,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    },
                    "model": model_name
                }
            except requests.exceptions.HTTPError as e:
                # Try to get error details from response
                try:
                    error_detail = r.json()
                    error_msg = f"{str(e)}: {error_detail.get('error', {}).get('message', 'No details')}"
                except:
                    error_msg = str(e)
                last_err = Exception(error_msg)
                if attempt < retries:
                    sleep_s = min(30, (2 ** attempt) + 0.5)  # Fixed jitter instead of random
                    time.sleep(sleep_s)
                else:
                    raise last_err
            except Exception as e:
                last_err = e
                if attempt < retries:
                    sleep_s = min(30, (2 ** attempt) + 0.5)  # Fixed jitter instead of random
                    time.sleep(sleep_s)
                else:
                    raise
        
        raise last_err or RuntimeError("Unknown error")
