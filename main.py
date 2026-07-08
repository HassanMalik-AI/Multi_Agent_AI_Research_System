# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# import your existing logic
from core_agents.pipeline import run_pipeline # change to your actual function name
# from core_agents.agents import researcher, analyzer... if you call them directly

app = FastAPI()

# Serve the HTML/CSS/JS
app.mount("/static", StaticFiles(directory="Agent_frontend"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("Agent_frontend/home.html")

@app.get("/{page}.html")
async def serve_pages(page: str):
    file_path = f"Agent_frontend/{page}.html"
    if os.path.exists(file_path):
        return FileResponse(file_path)
        
    return FileResponse("Agent_frontend/home.html")

class ResearchRequest(BaseModel):
    query: str

@app.post("/api/research")
async def research(req: ResearchRequest):
    # call your multi-agent pipeline
    result = await run_pipeline(req.query) # if it's sync, remove await
    # result should be dict like {"summary": "...", "agents": {...}}
    return result