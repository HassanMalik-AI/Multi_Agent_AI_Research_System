from core_agents.agents import search_agent,reader_agent,writer_chain,checking_chain
import asyncio

async def run_pipeline(topic:str):
    print("Starting research pipline...")
    print("Phase 1: Search")
    
    search_result = await search_agent().ainvoke({"topic": topic})
    print(search_result)

    print("Phase 2: Reading")
    reader_result = await reader_agent().ainvoke({"topic": topic})
    print(reader_result)

    print("Phase 3: Writing")
    writer_result = await writer_chain.ainvoke({"topic": topic, "research": search_result})
    print(writer_result)

    print("Phase 4: Checking")
    checking_result = checking_chain.invoke({"topic": topic, "report": writer_result})
    print(checking_result)

    return writer_result

if __name__ == "__main__":
    topic = input("Enter the topic to research: ")
    result = asyncio.run(run_pipeline(topic))
    print(result)
