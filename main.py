import os

from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import x_search

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600,  # Override default timeout with longer timeout for reasoning models
)
chat = client.chat.create(
    model="grok-4-1-fast",
    tools=[
        x_search(
            allowed_x_handles=["iScienceLuvr", "arankomatsuzaki", "rohanpaul_ai"],
            enable_image_understanding=True,
            enable_video_understanding=True,
        )
    ],
)
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("Can you find me some recent AI papers?"))
response = chat.sample()
print(response.content)
