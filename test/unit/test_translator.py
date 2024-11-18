from unittest.mock import patch
from src.translator import translate_content, query_llm_robust
import openai

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

@patch.object(openai.ChatCompletion, 'create')
def test_llm_normal_response(mocker):
    """
    Test for a normal response from the LLM where the input is translated properly.
    """
    # Mock the model's response to return a valid translation
    mocker.return_value.choices[0].message.content = "The translation is: This is your first example."

    # Assert that query_llm_robust handles valid translation correctly
    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (False, "This is your first example."), "Failed to handle normal response correctly"

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response(mocker):
    """
    Test for handling a gibberish response from the LLM.
    """
    # Mock the model's response to return 'Unintelligible'
    mocker.return_value.choices[0].message.content = "Unintelligible"

    # Assert that query_llm_robust handles gibberish input gracefully
    result = query_llm_robust("asdkjhf aksdfjhas dfkljha sd")
    assert result == (False, "Unintelligible"), "Failed to handle gibberish response correctly"

@patch.object(openai.ChatCompletion, 'create')
def test_llm_unexpected_language(mocker):
    """
    Test for an unexpected response format from the LLM.
    """
    # Mock the model's response to return an unexpected message
    mocker.return_value.choices[0].message.content = "I don't understand your request"

    # Assert that query_llm_robust handles this gracefully
    result = query_llm_robust("Bonjour tout le monde!")
    assert result == (False, "Error: Unexpected response format"), "Failed to handle unexpected language response"

@patch.object(openai.ChatCompletion, 'create')
def test_llm_api_error(mocker):
    """
    Test for handling an API error gracefully.
    """
    # Simulate an API error by raising an exception in the mock
    mocker.side_effect = Exception("API request failed")

    # Assert that query_llm_robust returns a safe fallback response
    result = query_llm_robust("今日はとてもいい天気ですね。")
    assert result == (False, "Error: Unable to process request"), "Failed to handle API error gracefully"