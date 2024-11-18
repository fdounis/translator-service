import re
import openai
import os


def extract_translation(response_text: str) -> str:
    """
    Extracts the translated portion of the model's response, removing any introductory comments.
    Assumes that the translation appears after phrases like 'means:' or 'is:'.

    Args:
        response_text (str): The text response from the model.

    Returns:
        str: The extracted translation or the original text if no match is found.
    """
    match = re.search(r'(?:means:|is:|translation:|to English is:)\s*["\']?(.+?)["\']?$', response_text, re.IGNORECASE)
    return match.group(1) if match else response_text


def translate_content(post: str) -> tuple[bool, str]:
    '''
    Robust version of query_llm that handles unexpected model responses gracefully.
    '''

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    # Initial prompt to detect language and translate if needed
    prompt = (
        "Identify the language of the following text and translate it to English if it is not already in English. "
        "If the text is in English, return it as is. If the text is unintelligible or malformed, return 'Unintelligible'."
        f"\n\nText: \"{post}\""
    )

    try:
        # Call the OpenAI API for language detection and translation
        response = openai.ChatCompletion.create(
            engine="OPENAI_DEPLOYMENT_NAME",  # Replace with your specific model deployment name
            messages=[
                {"role": "system", "content": "You are an assistant trained to identify and translate text."},
                {"role": "user", "content": prompt}
            ]
        )

        # Ensure the response has the expected structure
        if not response.choices or not response.choices[0].message.content:
            print("Received empty or malformed response from the LLM.")
            return (False, "Error: Unexpected response format")

        # Extract the response text from the API response
        llm_output = response.choices[0].message.content.strip()

        # Check for phrases indicating unexpected response format
        if any(phrase in llm_output.lower() for phrase in ["i don't understand", "unrecognized format", "error"]):
            return (False, "Error: Unexpected response format")

        # Check if the response indicates the text is unintelligible
        if llm_output.lower() == "unintelligible":
            return (False, "Unintelligible")

        # Check if the response explicitly states the text is in English
        is_english = bool(re.search(r'\b(already in English|text is in English)\b', llm_output, re.IGNORECASE))

        # If no clear indication of English, check if the response text matches the input
        if not is_english:
            is_english = llm_output == post

        # Extract the actual translation if the text is not in English
        translation = extract_translation(llm_output) if not is_english else post

        # Final check to ensure output format is as expected
        if isinstance(is_english, bool) and isinstance(translation, str):
            return (is_english, translation)
        else:
            print(f"Unexpected format from LLM: {llm_output}")
            return (False, "Error: Unexpected response format")

    except Exception as e:
        # Log the error and return a fallback response to avoid breaking NodeBB
        print(f"Error during LLM query: {e}")
        return (False, "Error: Unable to process request")
