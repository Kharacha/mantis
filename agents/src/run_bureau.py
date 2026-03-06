from uagents import Bureau
from src.agents.search_agent import agent as search_agent, print_inspector_link

# Bureau hosts all agents on a single HTTP server (default below).
BUREAU_URI = "http://127.0.0.1:8000"

if __name__ == "__main__":
    # Print the Inspector link that uses the Bureau URI + the agent's DID
    print("\n--- Agent Inspector (Bureau mode) ---")
    print_inspector_link(bureau_uri=BUREAU_URI)
    print("-------------------------------------\n")

    bureau = Bureau(port=8000)  # this is the ONE server for all agents
    bureau.add(search_agent)

    bureau.run()  # CTRL+C to stop
