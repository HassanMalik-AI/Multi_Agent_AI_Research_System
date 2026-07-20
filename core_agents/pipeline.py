from core_agents.agents import search_agent,reader_agent,writer_chain,checking_chain
import asyncio

async def run_pipeline(topic:str) -> dict:
    print("Starting research pipline...")
    print("Phase 1: Search")
    
    search_result = await search_agent().ainvoke({"topic": topic})
    print(search_result)

    print("Phase 2: Reading")
    reader_result = await reader_agent().ainvoke({"topic": topic, "search_result": search_result})
    print(reader_result)

    print("Phase 3: Writing")
    writer_result = await writer_chain.ainvoke({"topic": topic, "research": reader_result})
    print(writer_result)

    print("Phase 4: Checking")
    checking_result = await checking_chain.ainvoke({"topic": topic, "report": writer_result})
    print(checking_result)

    return {
        "summary": writer_result,
        "agents": {
            "search": search_result,
            "reader": reader_result,
            "checker": checking_result
        }
    }

if __name__ == "__main__":
    topic = input("Enter the topic to research: ")
    result = asyncio.run(run_pipeline(topic))
    print(result)
