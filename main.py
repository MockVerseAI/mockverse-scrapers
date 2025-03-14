from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from company_research import main as research_company

app = FastAPI()

# Define a Pydantic model for the request body
class ResearchRequest(BaseModel):
    company_name: str

@app.post("/research")
async def trigger_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Endpoint to trigger company research asynchronously."""
    # Call the main function with the company name
    background_tasks.add_task(research_company, request.company_name)
    return {"message": "Research task started for company: " + request.company_name}

# To run the server, use the command: uvicorn main:app --reload 