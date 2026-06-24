# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# import your existing logic
from pipeline import run_pipeline # change to your actual function name
# from agents import researcher, analyzer... if you call them directly

app = FastAPI()

# Serve the HTML/CSS/JS
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("frontend/index.html")

class ResearchRequest(BaseModel):
    query: str

@app.post("/api/research")
async def research(req: ResearchRequest):
    # call your multi-agent pipeline
    result = await run_pipeline(req.query) # if it's sync, remove await
    # result should be dict like {"summary": "...", "agents": {...}}
    return result