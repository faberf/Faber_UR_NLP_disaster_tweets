import lmql

import os
import urllib.request


# def download_gguf(model_url, filename):
#     if not os.path.isfile(filename):
#         urllib.request.urlretrieve(model_url, filename)
#         print("file has been downloaded successfully")
#     else:
#         print("file already exists")

# download_gguf(
#     "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf", 
#     "zephyr-7b-beta.Q4_K_M.gguf"
# )

# download_gguf(
#     "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf", 
#     "llama-2-7b.Q4_K_M.gguf"
# )


m : lmql.LLM = lmql.model("gpt2-medium", endpoint="localhost:9999")


@lmql.query
def capital_sights(country):
    '''lmql
    "Q: What is the captital of {country}? \\n"
    "A: [CAPITAL] \\n"
    "Q: What is the main sight in {CAPITAL}? \\n"
    "A: [ANSWER]" where (len(TOKENS(CAPITAL)) < 10) and (len(TOKENS(ANSWER)) < 100) \
        and STOPS_AT(CAPITAL, '\\n') and STOPS_AT(ANSWER, '\\n')

    # return just the ANSWER 
    return ANSWER
    '''
    

print(capital_sights(country="the United Kingdom", model=m))