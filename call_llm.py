import PIL.Image
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
import json
import os
import re
import base64
from loguru import logger
from google.generativeai.types import HarmCategory, HarmBlockThreshold

AI_3RD_PROVIDERS = ["openai", "gemini"]


class LLMContentGenerator:

    def __call_openai(self, system_prompt: str, user_prompt: str, model: str, retry: int = 3, json: bool = True,
                      image_path: str = None, temperature: float = 0.0):

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            show_log(message=f"__call_openai", level="info")
            if image_path:
                base64_image = encode_image(image_path)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url":
                            {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                         }
                    ]}
                ]
                if json:
                    response = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        messages=messages,
                        temperature=temperature,
                        timeout=60 * 10  # 5 minutes
                    )
                    result = convert_prompt_to_json(response.choices[0].message.content)
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=60 * 10  # 5 minutes
                    )
                    result = response.choices[0].message.content

            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                if json:
                    response = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        messages=messages,
                        temperature=temperature,
                        timeout=60 * 5  # 5 minutes
                    )
                    result = convert_prompt_to_json(response.choices[0].message.content)
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=60 * 5  # 5 minutes
                    )
                    result = response.choices[0].message.content
                if len(result) == 0:
                    raise Exception("Empty response")
            return True, result, None

        except Exception as ex:

            if retry <= 0:
                show_log(message=f"Fail to call __call_openai with ex: {ex}, retry: {retry}", level="error")
                return False, None, str(ex)

            show_log(message=f"__call_openai -> Retry {retry}", level="error")

            retry -= 1

            return self.__call_openai(system_prompt=system_prompt, user_prompt=user_prompt, model=model, retry=retry,
                                      json=json, image_path=image_path, temperature=temperature)


    def __call_gemini(self, system_prompt: str, user_prompt: str, model: str, retry: int = 3, json: bool = True,
                      image_path: str = None, temperature: float = 0.0, top_k: int = 40,
                      top_p: float = 0.95):

        try:
            genai.configure(api_key="")

            show_log(message=f"__call_gemini", level="info")
            if json:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k if top_k < 40 else 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                }
            else:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k if top_k < 40 else 40,
                    "max_output_tokens": 8192
                }
            model_llm = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                system_instruction=system_prompt,
            )
            if image_path:
                with PIL.Image.open(image_path) as img:
                    message = [{'role': 'user', 'parts': [user_prompt, img]}]
                    response = model_llm.generate_content(message, request_options={'timeout': 600})
                    # model_llm.generate_content_async(message, request_options={'timeout': 600})
                if json:
                    result = convert_prompt_to_json(response.text)
                else:
                    result = response.text

            else:
                message = [{'role': 'user', 'parts': [user_prompt]}]
                responses = model_llm.generate_content(message, request_options={'timeout': 600})
                try:
                    full_response_text = ''
                    for response in responses:
                        if response.text:
                            full_response_text += response.text
                except Exception as e:
                    full_response_text = ''
                    for response in responses:
                        for part in response.parts:
                            full_response_text += part.text
                if json:
                    result = convert_prompt_to_json(responses.text)
                else:
                    result = full_response_text
                if len(result) == 0:
                    raise Exception("Empty response")
            return True, result, None

        except Exception as ex:

            if retry <= 0:
                show_log(message=f"Fail to call __call_gemini with ex: {ex}, retry: {retry}", level="error")
                return False, None, str(ex)

            show_log(message=f"__call_gemini -> Retry {retry}", level="error")

            retry -= 1

            return self.__call_gemini(system_prompt=system_prompt, user_prompt=user_prompt, model=model, retry=retry,
                                      json=json, image_path=image_path, temperature=temperature)

    def completion(
            self,
            system_prompt: str,
            user_prompt: str,
            providers: list[dict],
            json: bool = True,
            image_path: str = None,
    ):
        """
        Args: providers (list[dict]): List of providers to call
        Example: providers = [
                        {
                             "name": "openai",
                              "model": "gpt-4o",
                              "retry": 3,
                              "temperature": 0.0,
                              "top_k": 40,
                              "top_p": 0.95
                        }
                ]
        """
        if not providers:
            raise Exception("Providers is empty")
        try:
            is_success, response, error = False, None, None
            for provider in providers:
                if provider["name"] not in AI_3RD_PROVIDERS:
                    raise Exception(f"Provider {provider['name']} is not supported")
                if provider["name"] == "openai":
                    is_success, response, error = self.__call_openai(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=provider["model"],
                        retry=provider["retry"],
                        json=json,
                        image_path=image_path,
                        temperature=provider.get("temperature", 0.0)
                    )
                    if is_success:
                        return response

                if provider["name"] == "gemini":
                    is_success, response, error = self.__call_gemini(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=provider["model"],
                        retry=provider["retry"],
                        json=json,
                        image_path=image_path,
                        temperature=provider.get("temperature", 0.0),
                        top_k=provider.get("top_k", 40),
                        top_p=provider.get("top_p", 0.95),
                    )
                    if is_success:
                        return response
            if not is_success:
                raise Exception(error)

            return response
        except Exception as ex:
            show_log(message=ex, level="error")
            return None
            raise ex

def convert_prompt_to_json(presentation_json: str):
    json_content = ''
    try:
        # Find the start of the JSON content
        start_marker = '```json'
        start_index = presentation_json.find(start_marker)

        if start_index != -1:
            start_index += len(start_marker)  # Move past the '```json'

            # Find the end of the JSON content
            end_index = presentation_json.find('', start_index)

            if end_index != -1:
                # Extract the JSON content
                json_content = presentation_json[start_index:end_index].strip()
            else:
                json_content = presentation_json[start_index:].strip()

            # Debug print
            # print("Extracted JSON:", json_content)

            # Convert the string to a JSON object
            return json.loads(json_content)
        else:
            return json.loads(presentation_json)
    except json.JSONDecodeError as ex:
        # If there's a JSONDecodeError, attempt to fix
        pattern = r'",.*?[^\\]($|\n)'
        lines = presentation_json.split('\n')
        for i, line in enumerate(lines):
            # Find unterminated strings and attempt to close them
            matches = re.finditer(pattern, line, re.DOTALL)
            for match in matches:
                if match.group().endswith(('\n', '"')):
                    continue
                fixed_string = match.group()[:-1] + '"\n'
                line = line[:match.start()] + fixed_string + line[match.end():]
                lines[i] = line

        fixed_json_string = '\n'.join(lines)
        try:
            # Try parsing again after the fix
            return json.loads(fixed_json_string)
        except json.JSONDecodeError as e:
            # If still failing, return the error
            logger.error(f"Failed to auto-fix JSON: {e}")
            raise e


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def show_log(message, level: str = "info"):
    if level == "debug" and os.getenv('DEBUG'):
        logger.debug(str(message))
    elif level == "error":
        logger.error(str(message))
    else:
        logger.info(str(message))

