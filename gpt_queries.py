"""
None of the following stuff runs, they are here to document how we queried davinci.
"""
# we use the following function to prepare inputs for davinci
def pad_context(context_str):
    """
    insert newlines at desired location into context
    also pads a task descriptor for the underlying lm
    """
    task_descriptor = """Perform named entity recognition to extract movie titles. List all movie titles that appears in text above. Separate movie names using comma, and do not include years, genre, director, award and other movie-related word that are not titles."""
    out = 'Input text:\n'+context_str.replace('\\n', '')+'\nInstruction:\n'+task_descriptor+'\nExtracted Movie Titles:'
    return out
# ```
# The gpt completion query are sent using the following code snippet from Dylan Slack's OpenAI guide.
# ```
import sys
import time
def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='', n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(0.8)
    return response
```