import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
    model_kwargs={
        "device_map": "cuda",
    },
    pipeline_kwargs={
        "return_full_text": False,
        "max_new_tokens": 1024,
        "clean_up_tokenization_spaces": False,
    }
)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful, friendly, and knowledgeable AI assistant.

Your responsibilities:
- Answer the user's questions clearly and accurately.
- Keep answers concise unless the user asks for detailed explanations.
- If you don't know something, say so instead of making up information.
- Use simple language when explaining technical concepts.
- Break complex problems into logical steps.
- Be polite and conversational.
"""
    ),
    ("placeholder", "{history}"),
    ("human", "{user_input}")
])

message_history = []

print(".............llm................")
print("enter 0 to exit")

while True:
    inp = input("you: ")

    if inp == "0":
        break

    message_history.append(
        HumanMessage(content=inp)
    )

    prompt = prompt_template.invoke({
        "history": message_history,
        "user_input": inp
    })

    response = model.invoke(prompt)

    print(response.content)

    message_history.append(
        AIMessage(content=response.content)
    )