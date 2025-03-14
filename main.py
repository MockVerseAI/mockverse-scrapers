from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import asyncio
import os
import time
from uuid import uuid4

from company_research import main as research_company
from interview_prep_roadmap import InterviewPrepEnhancer

# Create FastAPI application
app = FastAPI(
    title="AI-Powered Research and Interview Preparation API",
    description="API for company research and personalized interview preparation roadmaps",
    version="1.0.0"
)

# Initialize the interview prep enhancer
enhancer = InterviewPrepEnhancer()

# Request models
class ResearchRequest(BaseModel):
    company_name: str

class InterviewPrepRequest(BaseModel):
    company_name: str
    job_role: str
    job_description: str
    days_until_interview: int
    resume_text: str
    
    class Config:
        schema_extra = {
            "example": {
                "company_name": "Google",
                "job_role": "Senior Software Engineer",
                "job_description": "We are looking for a Senior Software Engineer with 5+ years of experience in Python and cloud technologies. The ideal candidate will have experience with distributed systems, microservices architecture, and excellent problem-solving skills.",
                "days_until_interview": 7,
                "resume_text": "Experienced software engineer with 6 years of Python development and AWS cloud infrastructure expertise. Strong background in distributed systems and API development."
            }
        }

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: str

# Store for background tasks
background_tasks_status = {}

# Original company research endpoint
@app.post("/research", response_model=Dict[str, Any])
async def trigger_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger company research asynchronously.
    
    This endpoint starts a background task to research information about a company.
    """
    # Generate a unique task ID
    task_id = str(uuid4())
    
    # Create a wrapper function to track the task status
    async def research_task():
        try:
            background_tasks_status[task_id] = {"status": "in_progress", "type": "research"}
            result = await research_company(request.company_name)
            background_tasks_status[task_id] = {
                "status": "completed", 
                "result": {"company_name": request.company_name, "summary": "Research completed successfully"},
                "timestamp": __import__('time').time(),
                "type": "research"
            }
        except Exception as e:
            background_tasks_status[task_id] = {
                "status": "failed",
                "error": str(e),
                "timestamp": __import__('time').time(),
                "type": "research"
            }
    
    # Add the task to background tasks
    background_tasks.add_task(research_task)
    
    return {
        "task_id": task_id,
        "message": f"Research task started for company: {request.company_name}",
        "status": "in_progress"
    }

# New interview preparation endpoint
@app.post("/prepare-interview", response_model=TaskStatus)
async def prepare_interview(request: InterviewPrepRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to generate an interview preparation roadmap asynchronously.
    
    This endpoint starts a background task to create a personalized interview preparation
    plan with web research on the company, job role, and required skills.
    """
    # Generate a unique task ID
    task_id = str(uuid4())
    
    # Create a function to run in the background
    async def interview_prep_task():
        try:
            background_tasks_status[task_id] = {"status": "in_progress", "type": "interview_prep"}
            
            result = await enhancer.generate_roadmap_api(
                company_name=request.company_name,
                job_role=request.job_role,
                job_description=request.job_description,
                days_until_interview=request.days_until_interview,
                resume_text=request.resume_text
            )
            
            background_tasks_status[task_id] = {
                "status": "completed",
                "result": result,
                "timestamp": time.time(),
                "type": "interview_prep"
            }
        except Exception as e:
            background_tasks_status[task_id] = {
                "status": "failed", 
                "error": str(e),
                "timestamp": time.time(),
                "type": "interview_prep"
            }
    
    # Add the task to the background tasks
    background_tasks_status[task_id] = {"status": "in_progress", "type": "interview_prep"}
    background_tasks.add_task(interview_prep_task)
    
    return {
        "task_id": task_id,
        "status": "in_progress",
        "message": "Interview preparation task started. You can check the status using the /task-status endpoint."
    }

@app.get("/task-status/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str):
    """
    Check the status of an asynchronous task.
    
    This endpoint can be used to check the status of both research and interview preparation tasks.
    """
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task_info = background_tasks_status[task_id]
    
    if task_info["status"] == "completed":
        return {
            "status": "completed",
            "task_type": task_info.get("type", "unknown"),
            "result": task_info["result"]
        }
    elif task_info["status"] == "failed":
        return {
            "status": "failed",
            "task_type": task_info.get("type", "unknown"),
            "error": task_info["error"]
        }
    else:
        return {
            "status": "in_progress",
            "task_type": task_info.get("type", "unknown"),
            "message": "Task is still in progress"
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI-Powered Research and Interview Preparation API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/research", "method": "POST", "description": "Start company research task"},
            {"path": "/prepare-interview", "method": "POST", "description": "Start interview preparation task"},
            {"path": "/task-status/{task_id}", "method": "GET", "description": "Check status of any task"},
            {"path": "/api/health", "method": "GET", "description": "Health check endpoint"}
        ],
        "docs_url": "/docs"
    }

# Cleanup function to remove old completed tasks
@app.on_event("startup")
async def setup_task_cleanup():
    async def cleanup_old_tasks():
        import time
        while True:
            current_time = time.time()
            to_remove = []
            
            for task_id, task_info in background_tasks_status.items():
                if task_info["status"] in ["completed", "failed"]:
                    if "timestamp" in task_info and (current_time - task_info["timestamp"]) > 3600:  # 1 hour
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del background_tasks_status[task_id]
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_tasks())

# Create required directories if they don't exist
os.makedirs(os.path.join("results", "interview_roadmap"), exist_ok=True)

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 

# //uvicorn main:app --reload 