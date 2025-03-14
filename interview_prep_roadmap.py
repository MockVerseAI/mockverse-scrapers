import asyncio
import os
import json
import re
import random
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, unquote
from datetime import datetime

from google import genai
from google.genai import types
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode

# Load environment variables
load_dotenv()

class InterviewPrepEnhancer:
    """Class to enhance interview prep roadmaps with web-scraped data."""
    
    def __init__(self):
        """Initialize the InterviewPrepEnhancer."""
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.model = "gemini-2.0-flash-lite"
        self.search_count = 0  # Counter for rate limiting
        
    async def _sleep_with_jitter(self, base_seconds=2.0):
        """Sleep with random jitter to avoid rate limiting."""
        jitter = random.uniform(0.5, 1.5)  # 50% below to 50% above base time
        sleep_time = base_seconds * jitter
        print(f"Sleeping for {sleep_time:.2f} seconds...")
        await asyncio.sleep(sleep_time)
        
    async def search_google(self, crawler, query, num_results=10):
        """Perform a Google search and return results."""
        print(f"Searching for: {query}")
        encoded_query = quote_plus(query)
        google_url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"
        
        try:
            # Simple search similar to company_research.py
            search_results = await crawler.arun(
                url=google_url,
                wait_for_selector="div.g",
                extract_links=True
            )
            
            urls = []
            
            # Try to extract from extracted_links
            if hasattr(search_results, 'extracted_links') and search_results.extracted_links:
                for link in search_results.extracted_links:
                    url = link.get('href', '')
                    if url and "google" not in url and url not in urls:
                        urls.append(url)
            # Fallback to markdown links
            elif hasattr(search_results, 'markdown'):
                markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', search_results.markdown)
                for text, url in markdown_links:
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    url = unquote(url)
                    if "google" not in url and url not in urls:
                        urls.append(url)
            
            print(f"Found {len(urls)} URLs for query: {query}")
            return urls
            
        except Exception as e:
            print(f"Error during search for '{query}': {e}")
            return []
    
    async def extract_page_content(self, crawler, url, max_depth=0):
        """Extract content from a page, similar to company_research.py approach."""
        print(f"Extracting content from: {url}")
        
        try:
            # Simple extraction similar to company_research.py
            result = await crawler.arun(
                url=url,
                max_depth=max_depth,
                extract_title=True,
                extract_text=True,
                extract_metadata=True
            )
            
            # Extract content from result
            title = result.title if hasattr(result, 'title') else "Unknown Title"
            
            # Try different content fields in order of preference
            if hasattr(result, 'text') and result.text:
                content = result.text
            elif hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
            elif hasattr(result, 'html') and result.html:
                content = result.html
            else:
                content = "No content extracted"
            
            # Get metadata if available
            metadata = {}
            if hasattr(result, 'metadata') and result.metadata:
                metadata = result.metadata
            
            return {
                "url": url,
                "title": title,
                "content": content[:15000],  # Limit content size
                "metadata": metadata
            }
        except Exception as e:
            print(f"Error extracting from {url}: {e}")
            return None
    
    async def scrape_job_information(self, job_title, company_name=None, skills=None, days_until_interview=7):
        """Scrape relevant information for job interview preparation."""
        print(f"Searching for information on {job_title} at {company_name if company_name else 'any company'}")
        print(f"Interview preparation time: {days_until_interview} days")
        
        start_time = time.time()
        
        search_results = {
            "job_descriptions": [],
            "required_skills": [],
            "interview_questions": [],
            "company_info": [],
            "industry_trends": [],
            "quick_prep_tips": []  # Category for short-term preparation
        }
        
        # Create a browser config for the crawler
        browser_config = BrowserConfig(
            headless=True,
            viewport_width=1280,
            viewport_height=1024,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            java_script_enabled=True,
            browser_type="chromium"
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # 1. Job descriptions and requirements
            job_queries = [
                f"{job_title} job description responsibilities",
                f"{job_title} required skills qualifications",
                f"what does a {job_title} do day to day"
            ]
            
            if company_name:
                job_queries.append(f"{company_name} {job_title} job")
                job_queries.append(f"{company_name} careers {job_title}")
            
            for query in job_queries:
                urls = await self.search_google(crawler, query, num_results=8)
                if not urls:
                    print(f"No search results found for query: {query}")
                    continue
                    
                # Process more results if we have a specific company
                max_results = 3 if company_name and company_name.lower() in query.lower() else 2
                
                for url in urls[:max_results]:
                    content = await self.extract_page_content(crawler, url)
                    if content:
                        search_results["job_descriptions"].append(content)
            
            # If we haven't got any job descriptions, try alternative queries
            if not search_results["job_descriptions"]:
                print("Having trouble finding job descriptions. Trying alternative queries...")
                alt_queries = [
                    f"{job_title} job posting",
                    f"how to become a {job_title}",
                    f"{job_title} career path"
                ]
                
                for query in alt_queries:
                    urls = await self.search_google(crawler, query, num_results=8)
                    for url in urls[:3]:
                        content = await self.extract_page_content(crawler, url)
                        if content:
                            search_results["job_descriptions"].append(content)
            
            # 2. Interview questions
            question_queries = [
                f"{job_title} common interview questions",
                f"{job_title} technical interview questions",
                f"{job_title} behavioral interview questions"
            ]
            
            if company_name:
                question_queries.append(f"{company_name} {job_title} interview questions")
                question_queries.append(f"{company_name} interview process")
            
            for query in question_queries:
                urls = await self.search_google(crawler, query, num_results=8)
                for url in urls[:3]:
                    content = await self.extract_page_content(crawler, url)
                    if content:
                        search_results["interview_questions"].append(content)
            
            # 3. Skills and technologies
            if skills:
                for skill in skills[:5]:
                    skill_query = f"{skill} for {job_title} interview preparation"
                    urls = await self.search_google(crawler, skill_query, num_results=5)
                    for url in urls[:2]:
                        content = await self.extract_page_content(crawler, url)
                        if content:
                            search_results["required_skills"].append(content)
            else:
                skill_queries = [
                    f"{job_title} key skills",
                    f"{job_title} technologies",
                    f"{job_title} technical requirements"
                ]
                
                for query in skill_queries:
                    urls = await self.search_google(crawler, query, num_results=5)
                    for url in urls[:2]:
                        content = await self.extract_page_content(crawler, url)
                        if content:
                            search_results["required_skills"].append(content)
            
            # 4. Company information (if provided)
            if company_name:
                company_queries = [
                    f"{company_name} about",
                    f"{company_name} culture values",
                    f"{company_name} products services"
                ]
                
                for query in company_queries:
                    urls = await self.search_google(crawler, query, num_results=5)
                    for url in urls[:2]:
                        content = await self.extract_page_content(crawler, url)
                        if content:
                            search_results["company_info"].append(content)
            
            # 5. Industry trends
            trend_queries = [
                f"{job_title} industry trends {datetime.now().year}",
                f"{job_title} career path growth",
                f"{job_title} latest technologies"
            ]
            
            for query in trend_queries:
                urls = await self.search_google(crawler, query, num_results=5)
                for url in urls[:2]:
                    content = await self.extract_page_content(crawler, url)
                    if content:
                        search_results["industry_trends"].append(content)
            
            # 6. Quick preparation tips (especially useful for short-term preparation)
            if days_until_interview <= 7:
                quick_prep_queries = [
                    f"how to prepare for {job_title} interview in {days_until_interview} days",
                    f"last minute {job_title} interview preparation",
                    f"quick tips for {job_title} interview"
                ]
                
                for query in quick_prep_queries:
                    urls = await self.search_google(crawler, query, num_results=5)
                    for url in urls[:2]:
                        content = await self.extract_page_content(crawler, url)
                        if content:
                            search_results["quick_prep_tips"].append(content)
        
        # Log total scraping time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total scraping time: {elapsed_time:.2f} seconds")
        
        # Add fallback data if needed
        if not search_results["job_descriptions"]:
            print("No job descriptions found. Adding fallback data.")
            search_results["job_descriptions"].append({
                "url": "fallback",
                "title": f"{job_title} Job Description",
                "content": f"This is a fallback description for {job_title} position. The web scraping encountered issues."
            })
        
        if not search_results["interview_questions"]:
            print("No interview questions found. Adding fallback data.")
            search_results["interview_questions"].append({
                "url": "fallback",
                "title": f"{job_title} Interview Questions",
                "content": f"Common interview questions for {job_title} positions include technical and behavioral questions."
            })
        
        # Print data collection report
        for category, items in search_results.items():
            print(f"{category.replace('_', ' ').title()}: {len(items)} sources")
        
        return search_results
    
    def create_enhanced_prompt(self, user_input, scraped_data, days_until_interview=7):
        """Create an enhanced prompt with scraped data context."""
        # Extract key information from scraped data
        job_info = "\n".join([item["content"][:1000] for item in scraped_data["job_descriptions"]])
        question_info = "\n".join([item["content"][:1000] for item in scraped_data["interview_questions"]])
        skill_info = "\n".join([item["content"][:1000] for item in scraped_data["required_skills"]])
        company_info = "\n".join([item["content"][:1000] for item in scraped_data["company_info"]])
        trend_info = "\n".join([item["content"][:1000] for item in scraped_data["industry_trends"]])
        quick_prep_info = "\n".join([item["content"][:1000] for item in scraped_data.get("quick_prep_tips", [])])
        
        # Create an enhanced prompt with the scraped context
        enhanced_prompt = f"""
        USER INPUT:
        {user_input}
        
        PREPARATION TIMEFRAME: {days_until_interview} days until the interview
        
        ADDITIONAL CONTEXT FROM WEB RESEARCH:
        
        JOB DESCRIPTION INSIGHTS:
        {job_info}
        
        COMMON INTERVIEW QUESTIONS:
        {question_info}
        
        KEY SKILLS AND TECHNOLOGIES:
        {skill_info}
        
        {"COMPANY INFORMATION:" if company_info else ""}
        {company_info}
        
        INDUSTRY TRENDS:
        {trend_info}
        
        {"QUICK PREPARATION TIPS:" if quick_prep_info else ""}
        {quick_prep_info}
        
        Based on the user input and the additional context provided above, create a detailed and personalized interview preparation roadmap.
        The candidate has {days_until_interview} days until the interview, so please adjust the daily plan accordingly.
        Focus on addressing the specific needs mentioned in the user input, while incorporating insights from the researched information.
        Ensure the daily plan is realistic and achievable within the timeframe available.
        """
        
        return enhanced_prompt
    
    async def generate_enhanced_roadmap(self, user_input):
        """Generate an enhanced interview preparation roadmap."""
        # Parse user input to extract job title, company, and skills
        job_title, company_name, skills, days_until_interview = self.parse_user_input(user_input)
        
        try:
            # Add an overall timeout for the entire generation process
            async with asyncio.timeout(1800):  # 30 minute timeout for the entire process
                # Scrape relevant information
                print("Starting web research for enhanced context...")
                scraped_data = await self.scrape_job_information(job_title, company_name, skills, days_until_interview)
                
                # Create enhanced prompt with scraped context
                enhanced_prompt = self.create_enhanced_prompt(user_input, scraped_data, days_until_interview)
                
                # Generate the roadmap using the enhanced prompt
                print("\nGenerating enhanced interview preparation roadmap...")
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=enhanced_prompt),
                        ],
                    ),
                ]
                
                # Define generation config with schema
                generate_content_config = types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["summary", "focusAreas", "dailyPlan"],
                        properties={
                            "summary": genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                required=["overview", "keyStrengths", "developmentAreas"],
                                properties={
                                    "overview": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                        description="Brief overview of the preparation roadmap",
                                    ),
                                    "keyStrengths": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        description="Candidate's key strengths relevant to the job",
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                    "developmentAreas": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        description="Areas where the candidate should focus preparation",
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                },
                            ),
                            "focusAreas": genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                required=["technical", "behavioral", "companyKnowledge"],
                                properties={
                                    "technical": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        description="Technical skills to focus on during preparation",
                                        items=genai.types.Schema(
                                            type=genai.types.Type.OBJECT,
                                            required=["skill", "priority", "resources"],
                                            properties={
                                                "skill": genai.types.Schema(
                                                    type=genai.types.Type.STRING,
                                                    description="Name of the technical skill",
                                                ),
                                                "priority": genai.types.Schema(
                                                    type=genai.types.Type.STRING,
                                                    description="Priority level for preparation",
                                                    enum=["High", "Medium", "Low"],
                                                ),
                                                "resources": genai.types.Schema(
                                                    type=genai.types.Type.ARRAY,
                                                    description="Recommended resources for preparation",
                                                    items=genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                    ),
                                                ),
                                            },
                                        ),
                                    ),
                                    "behavioral": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        description="Behavioral competencies to focus on",
                                        items=genai.types.Schema(
                                            type=genai.types.Type.OBJECT,
                                            required=["competency", "sampleQuestions"],
                                            properties={
                                                "competency": genai.types.Schema(
                                                    type=genai.types.Type.STRING,
                                                    description="Name of the behavioral competency",
                                                ),
                                                "sampleQuestions": genai.types.Schema(
                                                    type=genai.types.Type.ARRAY,
                                                    description="Sample questions related to this competency",
                                                    items=genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                    ),
                                                ),
                                            },
                                        ),
                                    ),
                                    "companyKnowledge": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        description="Areas of company knowledge to research",
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                },
                            ),
                            "dailyPlan": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                description="Day-by-day preparation plan",
                                items=genai.types.Schema(
                                    type=genai.types.Type.OBJECT,
                                    required=["day", "focusTheme", "activities", "resources", "outcomes"],
                                    properties={
                                        "day": genai.types.Schema(
                                            type=genai.types.Type.INTEGER,
                                            description="Day number in the preparation timeline",
                                        ),
                                        "focusTheme": genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                            description="Main theme or focus for this day",
                                        ),
                                        "activities": genai.types.Schema(
                                            type=genai.types.Type.ARRAY,
                                            description="Specific activities planned for the day",
                                            items=genai.types.Schema(
                                                type=genai.types.Type.OBJECT,
                                                required=["title", "description", "timeAllocation", "priority"],
                                                properties={
                                                    "title": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Title of the activity",
                                                    ),
                                                    "description": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Detailed description of the activity",
                                                    ),
                                                    "timeAllocation": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Recommended time to spend on this activity",
                                                    ),
                                                    "priority": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Priority level of this activity",
                                                        enum=["High", "Medium", "Low"],
                                                    ),
                                                },
                                            ),
                                        ),
                                        "resources": genai.types.Schema(
                                            type=genai.types.Type.ARRAY,
                                            description="Resources to use for this day's activities",
                                            items=genai.types.Schema(
                                                type=genai.types.Type.OBJECT,
                                                required=["title", "type", "description"],
                                                properties={
                                                    "title": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Title of the resource",
                                                    ),
                                                    "type": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Type of resource",
                                                        enum=["Article", "Video", "Book", "Practice Exercise", "Tool", "Website", "Other"],
                                                    ),
                                                    "description": genai.types.Schema(
                                                        type=genai.types.Type.STRING,
                                                        description="Brief description of the resource",
                                                    ),
                                                },
                                            ),
                                        ),
                                        "outcomes": genai.types.Schema(
                                            type=genai.types.Type.ARRAY,
                                            description="Expected outcomes from completing this day's activities",
                                            items=genai.types.Schema(
                                                type=genai.types.Type.STRING,
                                            ),
                                        ),
                                    },
                                ),
                            ),
                        },
                    ),
                )
                
                response_text = ""
                try:
                    # Add a timeout for LLM generation
                    async with asyncio.timeout(300):  # 5 minute timeout for LLM generation
                        for chunk in self.client.models.generate_content_stream(
                            model=self.model,
                            contents=contents,
                            config=generate_content_config,
                        ):
                            chunk_text = chunk.text if hasattr(chunk, 'text') else ""
                            print(chunk_text, end="")
                            response_text += chunk_text
                    
                    # Save the results
                    file_path = self.save_results(user_input, scraped_data, response_text, job_title, company_name, days_until_interview)
                    
                    # Parse the JSON for API response
                    response_data = json.loads(response_text)
                    
                    return {
                        "roadmap": response_data,
                        "metadata": {
                            "job_title": job_title,
                            "company_name": company_name,
                            "days_until_interview": days_until_interview,
                            "file_path": file_path
                        }
                    }
                except asyncio.TimeoutError:
                    print("\n⏱️ LLM generation timeout. The model is taking too long to respond.")
                    return {"error": "LLM generation timeout. Please try again later."}
                except json.JSONDecodeError as e:
                    print(f"\n🔴 JSON parsing error: {e}")
                    print("Response wasn't valid JSON. Returning raw response instead.")
                    return {
                        "error": "Failed to parse JSON response",
                        "raw_response": response_text,
                        "metadata": {
                            "job_title": job_title,
                            "company_name": company_name
                        }
                    }
                except Exception as e:
                    print(f"Error generating content: {e}")
                    return {"error": str(e)}
        except asyncio.TimeoutError:
            print("\n⏱️ Overall process timeout. The operation took too long to complete.")
            return {"error": "Process timeout. The operation took too long to complete."}
        except Exception as e:
            print(f"\n🔴 Critical error during roadmap generation: {e}")
            return {"error": f"Critical error: {str(e)}"}
    
    def parse_user_input(self, user_input):
        """Parse structured user input to extract job title, company name, and skills."""
        job_title = ""
        company_name = None
        skills = []
        days_until_interview = None
        
        # Parse structured input
        # Extract company name
        company_match = re.search(r'Company Name:\s*(.*?)(?:\n|$)', user_input, re.IGNORECASE)
        if company_match and company_match.group(1).strip():
            company_name = company_match.group(1).strip()
        
        # Extract job role/title
        job_match = re.search(r'Job Role:\s*(.*?)(?:\n|$)', user_input, re.IGNORECASE)
        if job_match and job_match.group(1).strip():
            job_title = job_match.group(1).strip()
        
        # Extract days until interview
        days_match = re.search(r'Days Until Interview:\s*(\d+)', user_input, re.IGNORECASE)
        if days_match:
            days_until_interview = int(days_match.group(1))
        
        # Extract potential skills from job description and resume
        job_desc_match = re.search(r'Job Description:\s*(.*?)(?:\n\s*\n|$)', user_input, re.DOTALL | re.IGNORECASE)
        job_description = job_desc_match.group(1).strip() if job_desc_match else ""
        
        resume_match = re.search(r'Candidate Information:\s*(.*?)(?:\n\s*\n|$)', user_input, re.DOTALL | re.IGNORECASE)
        resume_text = resume_match.group(1).strip() if resume_match else ""
        
        # Combine job description and resume for skills extraction
        combined_text = job_description + " " + resume_text
        
        # Common technical skills to look for
        skill_keywords = [
            # Programming Languages
            "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "go", "golang", "swift", "kotlin", "php", "scala", "rust", "r",
            
            # Web Technologies
            "html", "css", "react", "angular", "vue", "node", "express", "django", "flask", "spring", "asp.net", "jquery", "bootstrap",
            
            # Data Technologies
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis", "oracle", "sql server", "sqlite",
            "hadoop", "spark", "kafka", "elasticsearch", "tableau", "power bi", "machine learning", "data science", "artificial intelligence",
            "deep learning", "nlp", "natural language processing", "computer vision", "data mining", "data analysis", "data visualization",
            "statistics", "big data", "etl", "data warehouse", "data modeling",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "google cloud", "cloud computing", "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible",
            "devops", "sre", "site reliability", "monitoring", "logging", "infrastructure as code", "microservices", "serverless",
            
            # Mobile
            "android", "ios", "mobile development", "react native", "flutter", "xamarin", "swift", "objective-c",
            
            # Other Technical Skills
            "git", "github", "gitlab", "test-driven development", "tdd", "api", "rest", "graphql", "websocket", "json", "xml",
            "cybersecurity", "network security", "penetration testing", "encryption", "blockchain", "distributed systems",
            
            # Methodologies & Project Management
            "agile", "scrum", "kanban", "waterfall", "lean", "product management", "project management", "jira", "confluence", "trello",
            
            # Soft Skills
            "leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management", 
            "stakeholder management", "presentation", "negotiation", "conflict resolution"
        ]
        
        # Check for skills in combined text
        for skill in skill_keywords:
            # Look for whole word matches with word boundaries
            if re.search(r'\b' + re.escape(skill) + r'\b', combined_text.lower()):
                skills.append(skill)
        
        # If no job title found, use a default value
        if not job_title:
            job_title = "software engineer"  # Default value
            
        # Default days until interview if not provided
        if not days_until_interview:
            days_until_interview = 7  # Default value
            
        print(f"Parsed input - Job: {job_title}, Company: {company_name}, Days until interview: {days_until_interview}, Skills found: {len(skills)}")
        
        # Add days_until_interview to the return tuple
        return job_title, company_name, skills, days_until_interview
    
    def save_results(self, user_input, scraped_data, response, job_title, company_name, days_until_interview=7):
        """Save the results to files for reference and return the file path."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_slug = job_title.replace(' ', '_').lower()
        company_slug = company_name.replace(' ', '_').lower() if company_name else "no_company"
        
        # Use the specified folder structure
        base_folder = os.path.join("results", "interview_roadmap")
        os.makedirs(base_folder, exist_ok=True)
        
        # Create a subfolder with timestamp to avoid overwriting
        results_folder = os.path.join(base_folder, f"{job_slug}_at_{company_slug}_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)
        
        # Save user input
        with open(os.path.join(results_folder, "user_input.txt"), "w", encoding="utf-8") as f:
            f.write(user_input)
        
        # Save scraped data (split into separate files for better organization)
        scraped_data_folder = os.path.join(results_folder, "scraped_data")
        os.makedirs(scraped_data_folder, exist_ok=True)
        
        # Save each category of scraped data separately
        for category, items in scraped_data.items():
            if items:  # Only save if there's data
                with open(os.path.join(scraped_data_folder, f"{category}.json"), "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2)
        
        # Save full scraped data for reference
        with open(os.path.join(scraped_data_folder, "all_data.json"), "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2)
        
        # Save response to different formats
        json_file_path = os.path.join(results_folder, "roadmap.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(response)
        
        # Save a copy to the base folder with unique name for easy access
        base_json_path = os.path.join(base_folder, f"roadmap_{job_slug}_at_{company_slug}_{timestamp}.json")
        with open(base_json_path, "w", encoding="utf-8") as f:
            f.write(response)
        
        # Try to parse and save as markdown for easier reading
        try:
            response_data = json.loads(response)
            
            # Create a markdown version of the response
            markdown_content = f"# Interview Preparation Roadmap for {job_title} at {company_name}\n\n"
            markdown_content += f"*Preparation time: {days_until_interview} days*\n\n"
            
            # Overview
            if "summary" in response_data and "overview" in response_data["summary"]:
                markdown_content += f"## Overview\n\n{response_data['summary']['overview']}\n\n"
            
            # Key Strengths
            if "summary" in response_data and "keyStrengths" in response_data["summary"]:
                markdown_content += "## Key Strengths\n\n"
                for strength in response_data["summary"]["keyStrengths"]:
                    markdown_content += f"- {strength}\n"
                markdown_content += "\n"
            
            # Areas to Develop
            if "summary" in response_data and "developmentAreas" in response_data["summary"]:
                markdown_content += "## Areas to Focus On\n\n"
                for area in response_data["summary"]["developmentAreas"]:
                    markdown_content += f"- {area}\n"
                markdown_content += "\n"
            
            # Technical Focus Areas
            if "focusAreas" in response_data and "technical" in response_data["focusAreas"]:
                markdown_content += "## Technical Focus Areas\n\n"
                for skill in response_data["focusAreas"]["technical"]:
                    markdown_content += f"### {skill['skill']} (Priority: {skill['priority']})\n\n"
                    markdown_content += "**Resources:**\n"
                    for resource in skill["resources"]:
                        markdown_content += f"- {resource}\n"
                    markdown_content += "\n"
            
            # Behavioral Competencies
            if "focusAreas" in response_data and "behavioral" in response_data["focusAreas"]:
                markdown_content += "## Behavioral Competencies\n\n"
                for comp in response_data["focusAreas"]["behavioral"]:
                    markdown_content += f"### {comp['competency']}\n\n"
                    markdown_content += "**Sample Questions:**\n"
                    for question in comp["sampleQuestions"]:
                        markdown_content += f"- {question}\n"
                    markdown_content += "\n"
            
            # Company Knowledge
            if "focusAreas" in response_data and "companyKnowledge" in response_data["focusAreas"]:
                markdown_content += "## Company Knowledge Areas\n\n"
                for area in response_data["focusAreas"]["companyKnowledge"]:
                    markdown_content += f"- {area}\n"
                markdown_content += "\n"
            
            # Daily Plan
            if "dailyPlan" in response_data:
                markdown_content += "## Daily Preparation Plan\n\n"
                for day in response_data["dailyPlan"]:
                    markdown_content += f"### Day {day['day']}: {day['focusTheme']}\n\n"
                    
                    markdown_content += "#### Activities:\n\n"
                    for activity in day["activities"]:
                        markdown_content += f"**{activity['title']}** *(Priority: {activity['priority']}, Time: {activity['timeAllocation']})*\n\n"
                        markdown_content += f"{activity['description']}\n\n"
                    
                    markdown_content += "#### Resources:\n\n"
                    for resource in day["resources"]:
                        markdown_content += f"**{resource['title']}** ({resource['type']})\n\n"
                        markdown_content += f"{resource['description']}\n\n"
                    
                    markdown_content += "#### Expected Outcomes:\n\n"
                    for outcome in day["outcomes"]:
                        markdown_content += f"- {outcome}\n"
                    markdown_content += "\n"
            
            # Save the markdown file
            with open(os.path.join(results_folder, "roadmap.md"), "w", encoding="utf-8") as f:
                f.write(markdown_content)
        
        except Exception as e:
            print(f"Could not create markdown version: {e}")
        
        print(f"\nResults saved in folder: {results_folder}")
        
        return base_json_path

    async def generate_roadmap_api(self, company_name: str, job_role: str, job_description: str, 
                                days_until_interview: int, resume_text: str) -> Dict[str, Any]:
        """API-friendly method to generate interview roadmap."""
        # Format the input into a structured prompt
        structured_input = f"""
Company Name: {company_name}
Job Role: {job_role}
Job Description: {job_description}
Days Until Interview: {days_until_interview}

Candidate Information:
{resume_text}
"""

        print("\nAnalyzing information and preparing a tailored interview roadmap...")
        
        # Generate enhanced roadmap based on the structured input
        return await self.generate_enhanced_roadmap(structured_input)


async def main():
    """Main function to run the interview prep enhancer."""
    enhancer = InterviewPrepEnhancer()
    
    # Get structured input
    print("\n=== Interview Preparation Roadmap Generator ===\n")
    
    company_name = input("Company Name: ")
    job_role = input("Job Role: ")
    
    print("\nJob Description (type 'END' on a new line when finished):")
    job_description_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        job_description_lines.append(line)
    job_description = "\n".join(job_description_lines)
    
    days_until_interview = input("\nDays Until Interview: ")
    
    print("\nCandidate Resume/Information (type 'END' on a new line when finished):")
    resume_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        resume_lines.append(line)
    resume_text = "\n".join(resume_lines)
    
    # Use the API method directly
    result = await enhancer.generate_roadmap_api(
        company_name=company_name, 
        job_role=job_role, 
        job_description=job_description, 
        days_until_interview=int(days_until_interview) if days_until_interview.isdigit() else 7,
        resume_text=resume_text
    )
    
    print(f"\nRoadmap generated successfully. Check the results folder for details.")

if __name__ == "__main__":
    asyncio.run(main())
