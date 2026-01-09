from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

resp = llm.invoke([
    HumanMessage(content="Say hello in one word")
])

print(resp.content)
