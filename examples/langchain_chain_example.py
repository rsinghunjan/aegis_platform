  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
"""
Example: use the small Aegis "LangChain-like" adapter to run an agent that uses tools
and a simple memory store.

This example is synchronous and intended to be run locally against a running Aegis API.

Before running:
- Ensure Aegis API is reachable via AEGIS_API_BASE and AEGIS_API_TOKEN env vars (or defaults).
- Ensure your model (DEFAULT_MODEL / DEFAULT_MODEL_VERSION) is registered and can accept the instruction template.

Usage:
  python examples/langchain_chain_example.py
"""
from api.langchain_adapter import AegisLLM, AgentRunner, register_tool
from api.memory import ConversationMemory

# Register a simple tool
@register_tool("get_time", "Return the current server time in ISO format")
def get_time(_inp: str) -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

@register_tool("echo", "Echo back the input")
def echo(inp: str) -> str:
    return f"echo:{inp}"

def main():
    llm = AegisLLM()  # uses env vars or defaults
    mem = ConversationMemory(capacity=50)

    # add example user turn to memory
    mem.add({"role": "user", "content": "You are helping a user with a scheduling question."})

    tools = []
    # load tools from registry (module-level)
    from api.langchain_adapter import _TOOL_REGISTRY
    for t in _TOOL_REGISTRY.values():
        tools.append(t)

    agent = AgentRunner(llm=llm, tools=tools, memory=mem, tenant_id="demo-tenant")
    prompt = "Please produce the current time and echo back 'hello' using tools. If you cannot, answer directly."

    resp = agent.run(prompt)
    print("AGENT RESPONSE:")
    print(resp)

if __name__ == "__main__":
    main()
