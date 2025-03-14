import asyncio
from crawl4ai import *
from google import genai
import re
from urllib.parse import unquote, urlparse, urljoin
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

async def useLLM(text):
    """Use Gemini model to process text."""
    try:
        print("Initializing Gemini client...")
        client = genai.Client(api_key=api_key)
        
        print("Sending request to Gemini model...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=text
        )
        
        print("Received response from Gemini model.")
        return response.text
    except Exception as e:
        print(f"Error in useLLM: {e}")
        return "Error generating summary"

def is_likely_company_website(url, company_name):
    """Check if URL is likely to be the company's official website."""
    # Parse the URL to get the domain
    domain = urlparse(url).netloc.lower()
    
    # Remove common TLDs and www
    domain = re.sub(r'^www\.', '', domain)
    domain = re.sub(r'\.(com|org|net|io|co|ai|app)$', '', domain)
    
    # Clean company name (lowercase, remove spaces and special chars)
    clean_company = re.sub(r'[^a-z0-9]', '', company_name.lower())
    clean_domain = re.sub(r'[^a-z0-9]', '', domain)
    
    # Check if company name is in the domain
    if clean_company in clean_domain or clean_domain in clean_company:
        return True
    
    # Check for common official site patterns
    official_patterns = [
        "official", "corp", "inc", "corporate", "group", 
        "global", "international", "worldwide", "company"
    ]
    
    for pattern in official_patterns:
        if pattern in domain:
            return True
            
    return False

def is_social_media(url):
    """Check if URL is a social media platform."""
    social_domains = [
        "linkedin.com", "facebook.com", "twitter.com", "instagram.com", 
        "youtube.com", "github.com", "medium.com", "tiktok.com"
    ]
    domain = urlparse(url).netloc.lower()
    return any(social in domain for social in social_domains)

def get_social_platform(url):
    """Extract which social media platform a URL belongs to."""
    domain = urlparse(url).netloc.lower()
    if "linkedin" in domain: return "LinkedIn"
    if "facebook" in domain: return "Facebook"
    if "twitter" in domain or "x.com" in domain: return "Twitter"
    if "instagram" in domain: return "Instagram"
    if "youtube" in domain: return "YouTube"
    if "github" in domain: return "GitHub"
    if "medium" in domain: return "Medium"
    if "tiktok" in domain: return "TikTok"
    return "Other"

def clean_url(url, base_url=None):
    """Clean and normalize a URL."""
    if not url:
        return None
    
    # Handle relative URLs
    if base_url and not (url.startswith('http://') or url.startswith('https://')):
        url = urljoin(base_url, url)
    
    # Remove fragments and most query params
    url = url.split('#')[0]
    if '?' in url:
        main_url = url.split('?')[0]
        # Keep only essential query params (if needed)
        url = main_url
    
    return url

async def search_google(crawler, query, num_results=10):
    """Perform a Google search and return results."""
    google_url = f"https://www.google.com/search?q={query}&num={num_results}"
    
    search_results = await crawler.arun(
        url=google_url,
        wait_for_selector="div.g",
        extract_links=True,
    )
    
    urls = []
    
    # Try to extract from links
    if hasattr(search_results, 'extracted_links') and search_results.extracted_links:
        for link in search_results.extracted_links:
            url = link.get('href', '')
            if url and "google" not in url and url not in urls:
                urls.append(url)
    else:
        # Fallback to markdown
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', search_results.markdown)
        for text, url in markdown_links:
            if url.startswith('/url?q='):
                url = url.split('/url?q=')[1].split('&')[0]
            url = unquote(url)
            if "google" not in url and url not in urls:
                urls.append(url)
    
    return urls

async def analyze_website_structure(website_data):
    """Analyze the structure of the website content."""
    structure = {
        "total_pages": 0,
        "sections": {},
        "content_types": [],
        "navigation": [],
        "main_topics": []
    }
    
    def analyze_page(page_data, depth=0):
        if not page_data:
            return
        
        structure["total_pages"] += 1
        
        # Analyze URL structure
        parsed_url = urlparse(page_data["url"])
        path_parts = parsed_url.path.strip("/").split("/")
        
        # Track sections based on URL structure
        if path_parts[0]:
            section = path_parts[0]
            if section not in structure["sections"]:
                structure["sections"][section] = []
            structure["sections"][section].append(page_data["url"])
        
        # Analyze navigation
        if depth == 0:  # Only for main pages
            for link in page_data.get("links", []):
                if link.get("text"):
                    structure["navigation"].append({
                        "text": link["text"],
                        "url": link["url"]
                    })
        
        # Extract main topics from content
        if page_data.get("content"):
            # Simple topic extraction from headers and prominent text
            headers = re.findall(r'<h[1-3][^>]*>(.*?)</h[1-3]>', page_data["content"])
            structure["main_topics"].extend(headers)
            structure["main_topics"] = list(set(structure["main_topics"]))
        
        # Recursively analyze sub-pages
        for sub_page in page_data.get("sub_pages", []):
            analyze_page(sub_page, depth + 1)
    
    analyze_page(website_data)
    return structure

async def extract_page_content(crawler, url, max_depth=2, extract_links=True, visited_urls=None):
    """Extract content from a page with recursive crawling."""
    if visited_urls is None:
        visited_urls = set()
    
    if url in visited_urls or not url:
        return None
    
    try:
        # Clean and validate URL
        clean_url = url.split('#')[0]  # Remove fragments
        if not clean_url.startswith(('http://', 'https://')):
            return None
        
        visited_urls.add(clean_url)
        print(f"Crawling: {clean_url}")
        
        result = await crawler.arun(
            url=clean_url,
            max_depth=0,  # We'll handle depth manually
            extract_title=True,
            extract_links=extract_links,
            preserve_images=True,
            extract_text=True,
            extract_metadata=True,
            timeout=30  # Add timeout to prevent hanging
        )
        
        # Extract basic info
        title = result.title if hasattr(result, 'title') else "Unknown Title"
        content = result.text if hasattr(result, 'text') else result.markdown
        
        # Extract metadata if available
        metadata = {}
        if hasattr(result, 'metadata'):
            metadata = result.metadata
        
        # Extract links if available
        links = []
        sub_pages = []
        
        if extract_links and max_depth > 0 and hasattr(result, 'extracted_links'):
            for link in result.extracted_links:
                link_url = link.get('href', '')
                link_text = link.get('text', '')
                if link_url:
                    full_url = urljoin(clean_url, link_url)
                    if is_same_domain(full_url, clean_url) and full_url not in visited_urls:
                        links.append({"url": full_url, "text": link_text})
                        # Recursively crawl internal links
                        sub_page = await extract_page_content(
                            crawler,
                            full_url,
                            max_depth - 1,
                            extract_links,
                            visited_urls
                        )
                        if sub_page:
                            sub_pages.append(sub_page)
        
        return {
            "url": clean_url,
            "title": title,
            "content": content,
            "metadata": metadata,
            "links": links,
            "sub_pages": sub_pages
        }
    except Exception as e:
        print(f"Error extracting from {url}: {e}")
        return None

async def find_company_pages(company_website_data, company_name):
    """Identify important company pages from the main website data."""
    important_pages = {
        "about": None,
        "products": None,
        "services": None,
        "team": None,
        "careers": None,
        "contact": None,
        "blog": None,
        "press": None,
        "investors": None,
        "partners": None,
    }
    
    # Keywords to identify page types
    page_keywords = {
        "about": ["about", "company", "who we are", "our story", "mission"],
        "products": ["products", "solutions", "offerings", "what we do", "software"],
        "services": ["services", "what we offer", "solutions", "consulting"],
        "team": ["team", "leadership", "management", "people", "executives", "board"],
        "careers": ["careers", "jobs", "join", "work with us", "opportunities"],
        "contact": ["contact", "reach", "get in touch", "locations", "offices"],
        "blog": ["blog", "news", "articles", "insights", "resources"],
        "press": ["press", "news", "media", "releases", "announcements"],
        "investors": ["investors", "investor relations", "shareholders", "financials"],
        "partners": ["partners", "ecosystem", "channel", "alliance", "resellers"],
    }
    
    # Check links for important pages
    for link in company_website_data.get("links", []):
        link_url = link.get("url", "")
        link_text = link.get("text", "").lower()
        
        # Skip external links
        if link_url and not is_same_domain(link_url, company_website_data["url"]):
            continue
            
        # Check each page type
        for page_type, keywords in page_keywords.items():
            if not important_pages[page_type]:  # Only if we haven't found this type yet
                if any(keyword in link_text for keyword in keywords):
                    important_pages[page_type] = link_url
                    break
    
    return important_pages

def is_same_domain(url1, url2):
    """Check if two URLs have the same domain."""
    try:
        domain1 = urlparse(url1).netloc
        domain2 = urlparse(url2).netloc
        return domain1 == domain2
    except:
        return False

def extract_contact_info(text):
    """Extract contact information from text."""
    contact_info = {
        "emails": [],
        "phones": [],
        "addresses": []
    }
    
    # Extract emails
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    contact_info["emails"] = list(set(emails))
    
    # Extract phone numbers (basic pattern)
    phones = re.findall(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)
    contact_info["phones"] = list(set(phones))
    
    return contact_info

async def find_social_media(crawler, company_name):
    """Find social media profiles for the company."""
    social_profiles = {}
    
    platforms = ["linkedin", "twitter", "facebook", "instagram", "youtube"]
    
    for platform in platforms:
        query = f"{company_name} official {platform}"
        results = await search_google(crawler, query, num_results=3)
        
        for url in results:
            if platform in url.lower() and not social_profiles.get(platform):
                social_profiles[platform] = url
                break
    
    return social_profiles

async def analyze_competitors(crawler, company_name, company_website=None):
    """Try to identify competitors of the company."""
    query = f"{company_name} competitors in industry"
    competitor_urls = await search_google(crawler, query, num_results=5)
    
    competitors = []
    for url in competitor_urls[:3]:  # Limit to top 3 results
        competitor_data = await extract_page_content(crawler, url, max_depth=0)
        if competitor_data:
            competitors.append({
                "source": competitor_data["title"],
                "url": url,
                "info": competitor_data["content"][:1000]  # Limit content size
            })
    
    return competitors

async def research_news(crawler, company_name):
    """Find recent news about the company."""
    query = f"{company_name} news recent"
    news_urls = await search_google(crawler, query, num_results=5)
    
    news_items = []
    for url in news_urls[:3]:  # Limit to top 3 results
        news_data = await extract_page_content(crawler, url, max_depth=0)
        if news_data:
            news_items.append({
                "headline": news_data["title"],
                "url": url,
                "summary": news_data["content"][:1000]  # Limit content size
            })
    
    return news_items

async def research_company(company_name):
    """Perform comprehensive company research and return structured data."""
    start_time = time.time()  # Start total time tracking
    print(f"\n🔍 Starting deep research on {company_name}...\n")
    
    research_data = {
        "company_name": company_name,
        "research_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "website_data": {
            "main_site": None,
            "structure": None,
            "pages": {},
            "extracted_info": {
                "products": [],
                "services": [],
                "team": [],
                "locations": [],
                "technologies": [],
                "contact": {},
                "careers": [],
                "partnerships": [],
                "press_releases": [],
                "case_studies": [],
                "technical_docs": [],
                "blog_posts": [],
                "events": []
            }
        },
        "social_presence": {},
        "news_and_updates": [],
        "industry_data": {},
        "financial_info": {},
        "cross_validation": {},
        "additional_sources": {}
    }
    
    async with AsyncWebCrawler() as crawler:
        # 1. Enhanced Website Analysis
        print("Step 1/8: Deep crawling company website...")
        search_query = f"{company_name} official website"
        company_urls = await search_google(crawler, search_query, num_results=15)  # Increased results
        
        official_website = None
        for url in company_urls:
            if is_likely_company_website(url, company_name):
                official_website = url
                break
        
        if official_website:
            print(f"Found official website: {official_website}")
            # Perform deeper crawl of the website
            website_content = await extract_page_content(
                crawler,
                official_website,
                max_depth=5,  # Increased depth
                extract_links=True
            )
            
            if website_content:
                research_data["website_data"]["main_site"] = website_content
                site_structure = await analyze_website_structure(website_content)
                research_data["website_data"]["structure"] = site_structure
                
                # Enhanced content processing
                def process_page_content(page):
                    if not page:
                        return
                    
                    page_url = page.get("url", "")
                    page_content = page.get("content", "")
                    page_title = page.get("title", "").lower()
                    page_metadata = page.get("metadata", {})
                    
                    # Store complete page data
                    research_data["website_data"]["pages"][page_url] = {
                        "title": page.get("title"),
                        "content": page_content,
                        "metadata": page_metadata,
                        "links": page.get("links", [])
                    }
                    
                    # Enhanced content categorization
                    categories = {
                        "products": ["product", "solution", "offering", "platform", "technology"],
                        "services": ["service", "consulting", "support", "professional"],
                        "team": ["team", "leadership", "management", "executive", "board", "director"],
                        "careers": ["career", "job", "position", "opening", "work with us", "join us"],
                        "partnerships": ["partner", "alliance", "ecosystem", "marketplace"],
                        "press_releases": ["press", "news", "announcement", "release"],
                        "case_studies": ["case study", "success story", "customer story"],
                        "technical_docs": ["documentation", "api", "developer", "tech spec"],
                        "blog_posts": ["blog", "article", "insight", "thought leadership"],
                        "events": ["event", "webinar", "conference", "meetup"]
                    }
                    
                    for category, keywords in categories.items():
                        if any(keyword in page_title or keyword in page_content.lower() for keyword in keywords):
                            research_data["website_data"]["extracted_info"][category].append({
                                "title": page.get("title"),
                                "url": page_url,
                                "content": page_content,
                                "metadata": page_metadata
                            })
                    
                    # Extract contact information
                    contact_info = extract_contact_info(page_content)
                    if contact_info["emails"] or contact_info["phones"]:
                        research_data["website_data"]["extracted_info"]["contact"][page_url] = contact_info
                    
                    # Process sub-pages
                    for sub_page in page.get("sub_pages", []):
                        process_page_content(sub_page)
                
                process_page_content(website_content)
        
        # 2. Enhanced Social Media Analysis
        print("Step 2/8: Deep social media analysis...")
        platforms = [
            "linkedin", "twitter", "facebook", "instagram", "youtube",
            "github", "medium", "crunchbase", "glassdoor", "indeed"
        ]
        
        social_profiles = {}
        for platform in platforms:
            query = f"{company_name} official {platform}"
            results = await search_google(crawler, query, num_results=5)
            
            for url in results:
                if platform in url.lower() and not social_profiles.get(platform):
                    # Crawl the social media page
                    social_content = await extract_page_content(crawler, url, max_depth=1)
                    if social_content:
                        social_profiles[platform] = {
                            "url": url,
                            "title": social_content.get("title", ""),
                            "content": social_content.get("content", ""),
                            "metadata": social_content.get("metadata", {})
                        }
                    break
        
        research_data["social_presence"] = social_profiles
        
        # 3. Enhanced News Analysis
        print("Step 3/8: Gathering comprehensive news coverage...")
        news_queries = [
            f"{company_name} news recent",
            f"{company_name} press release",
            f"{company_name} announcement",
            f"{company_name} funding",
            f"{company_name} acquisition"
        ]
        
        news_items = []
        for query in news_queries:
            news_urls = await search_google(crawler, query, num_results=10)
            for url in news_urls[:5]:
                news_data = await extract_page_content(crawler, url, max_depth=0)
                if news_data:
                    news_items.append({
                        "headline": news_data["title"],
                        "url": url,
                        "content": news_data["content"],
                        "metadata": news_data.get("metadata", {}),
                        "source_query": query
                    })
        
        research_data["news_and_updates"] = news_items
        
        # 4. Enhanced Industry Analysis
        print("Step 4/8: Deep industry analysis...")
        industry_queries = [
            f"{company_name} competitors",
            f"{company_name} market position",
            f"{company_name} industry analysis",
            f"{company_name} market share",
            f"{company_name} versus"
        ]
        
        industry_data = {
            "competitors": [],
            "market_analysis": [],
            "industry_trends": []
        }
        
        for query in industry_queries:
            results = await search_google(crawler, query, num_results=10)
            for url in results[:3]:
                content = await extract_page_content(crawler, url, max_depth=0)
                if content:
                    industry_data["market_analysis"].append({
                        "source": content["title"],
                        "url": url,
                        "content": content["content"],
                        "query_context": query
                    })
        
        # 5. Financial Information
        print("Step 5/8: Gathering financial information...")
        financial_queries = [
            f"{company_name} revenue",
            f"{company_name} funding",
            f"{company_name} investors",
            f"{company_name} valuation",
            f"{company_name} financial results"
        ]
        
        financial_info = {
            "funding_rounds": [],
            "investors": [],
            "financial_reports": [],
            "market_data": []
        }
        
        for query in financial_queries:
            results = await search_google(crawler, query, num_results=5)
            for url in results[:3]:
                content = await extract_page_content(crawler, url, max_depth=0)
                if content:
                    financial_info["market_data"].append({
                        "source": content["title"],
                        "url": url,
                        "content": content["content"],
                        "query_context": query
                    })
        
        research_data["financial_info"] = financial_info
        
        # 6. Technical Stack Analysis
        print("Step 6/8: Analyzing technical stack...")
        tech_queries = [
            f"{company_name} technology stack",
            f"{company_name} engineering blog",
            f"{company_name} github",
            f"{company_name} developer",
            f"{company_name} architecture"
        ]
        
        tech_info = []
        for query in tech_queries:
            results = await search_google(crawler, query, num_results=5)
            for url in results[:3]:
                content = await extract_page_content(crawler, url, max_depth=0)
                if content:
                    tech_info.append({
                        "source": content["title"],
                        "url": url,
                        "content": content["content"],
                        "query_context": query
                    })
        
        research_data["technical_stack"] = tech_info
        
        # 7. Employee Reviews and Culture
        print("Step 7/8: Analyzing company culture...")
        culture_queries = [
            f"{company_name} employee reviews",
            f"{company_name} work culture",
            f"{company_name} glassdoor",
            f"{company_name} benefits",
            f"{company_name} workplace"
        ]
        
        culture_info = []
        for query in culture_queries:
            results = await search_google(crawler, query, num_results=5)
            for url in results[:3]:
                content = await extract_page_content(crawler, url, max_depth=0)
                if content:
                    culture_info.append({
                        "source": content["title"],
                        "url": url,
                        "content": content["content"],
                        "query_context": query
                    })
        
        research_data["company_culture"] = culture_info
        
        # 8. Cross-validation and Additional Sources
        print("Step 8/8: Cross-validating information...")
        validation_sources = [
            "wikipedia",
            "crunchbase",
            "bloomberg",
            "reuters",
            "forbes"
        ]
        
        for source in validation_sources:
            query = f"{company_name} {source}"
            results = await search_google(crawler, query, num_results=3)
            if results:
                content = await extract_page_content(crawler, results[0], max_depth=0)
                if content:
                    research_data["cross_validation"][source] = {
                        "url": results[0],
                        "content": content["content"],
                        "title": content["title"]
                    }
    
    # Generate enhanced research summary
    print("\nGenerating comprehensive research summary...")
    llm_start_time = time.time()  # Start LLM timing
    
    summary_prompt = f"""
    Create an extremely detailed and comprehensive analysis of {company_name} based on the following extensive research data.
    Cross-validate information across multiple sources to ensure accuracy.
    
    MAIN WEBSITE CONTENT:
    {research_data["website_data"]["main_site"]["content"] if research_data["website_data"]["main_site"] else "No main site content available"}
    
    PRODUCTS AND SERVICES:
    {json.dumps([p["content"] for p in research_data["website_data"]["extracted_info"]["products"]], indent=2)}
    
    TEAM AND LEADERSHIP:
    {json.dumps([t["content"] for t in research_data["website_data"]["extracted_info"]["team"]], indent=2)}
    
    PARTNERSHIPS AND ECOSYSTEM:
    {json.dumps([p["content"] for p in research_data["website_data"]["extracted_info"]["partnerships"]], indent=2)}
    
    TECHNICAL DOCUMENTATION AND STACK:
    {json.dumps([t["content"] for t in research_data["website_data"]["extracted_info"]["technical_docs"]], indent=2)}
    
    CASE STUDIES AND SUCCESS STORIES:
    {json.dumps([c["content"] for c in research_data["website_data"]["extracted_info"]["case_studies"]], indent=2)}
    
    SOCIAL MEDIA PRESENCE:
    {json.dumps({k: v["content"] for k, v in research_data["social_presence"].items()}, indent=2)}
    
    NEWS AND DEVELOPMENTS:
    {json.dumps([n["content"] for n in research_data["news_and_updates"]], indent=2)}
    
    INDUSTRY ANALYSIS:
    {json.dumps([m["content"] for m in research_data["industry_data"].get("market_analysis", [])], indent=2)}
    
    FINANCIAL INFORMATION:
    {json.dumps([f["content"] for f in research_data["financial_info"].get("market_data", [])], indent=2)}
    
    COMPANY CULTURE:
    {json.dumps([c["content"] for c in research_data.get("company_culture", [])], indent=2)}
    
    CROSS-VALIDATION SOURCES:
    {json.dumps({k: v["content"] for k, v in research_data["cross_validation"].items()}, indent=2)}
    """
    
    research_summary = await useLLM(summary_prompt)
    llm_time = time.time() - llm_start_time  # Calculate LLM time
    
    total_time = time.time() - start_time  # Calculate total time
    
    return {
        "raw_research": research_data,
        "summary": research_summary,
        "timing": {
            "llm_time": llm_time,
            "total_time": total_time
        }
    }

async def main(company_name=None):
    """Gather comprehensive company information and store results."""
    if not company_name:
        # Get company name from user input if not provided
        company_name = input("Enter the company name to research: ")
    
    # Perform research
    print("\nStarting comprehensive company research...")
    research_results = await research_company(company_name)
    
    # Generate output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_folder_name = f"{company_name.replace(' ', '_')}_{timestamp}"
    results_folder = os.path.join("results", base_folder_name)
    os.makedirs(results_folder, exist_ok=True)  # Create the directory if it doesn't exist
    
    research_filename = os.path.join(results_folder, "research.json")
    debug_filename = os.path.join(results_folder, "debug_crawl.json")
    summary_filename = os.path.join(results_folder, "summary.md")
    
    print("\n" + "="*80)
    print(f"COMPANY RESEARCH: {company_name.upper()}")
    print("=" * 80 + "\n")
    
    print("RESEARCH SUMMARY:")
    print("-" * 40)
    print(research_results["summary"])
    
    # Display timing information
    llm_minutes = int(research_results["timing"]["llm_time"] // 60)
    llm_seconds = int(research_results["timing"]["llm_time"] % 60)
    total_minutes = int(research_results["timing"]["total_time"] // 60)
    total_seconds = int(research_results["timing"]["total_time"] % 60)
    
    print("\nTIMING INFORMATION:")
    print("-" * 40)
    print(f"LLM Summary Generation: {llm_minutes}m {llm_seconds}s")
    print(f"Total Research Time: {total_minutes}m {total_seconds}s")
    
    # Save raw research data to JSON file
    try:
        with open(research_filename, "w", encoding='utf-8') as f:
            json.dump(research_results["raw_research"], f, indent=2)
        print(f"\nDetailed research data saved to {research_filename}")
        
        # Create debug crawl data
        debug_data = {
            "company_name": company_name,
            "crawl_time": timestamp,
            "main_site": None,
            "crawled_pages": []
        }
        
        # Extract main site data
        if research_results["raw_research"]["website_data"]["main_site"]:
            main_site = research_results["raw_research"]["website_data"]["main_site"]
            debug_data["main_site"] = {
                "url": main_site["url"],
                "title": main_site["title"],
                "raw_content": main_site["content"],
                "metadata": main_site["metadata"]
            }
        
        # Extract all crawled pages
        def extract_page_data(page):
            page_data = {
                "url": page["url"],
                "title": page["title"],
                "raw_content": page["content"],
                "metadata": page["metadata"],
                "links": page["links"]
            }
            return page_data
        
        def process_pages(page):
            if page:
                debug_data["crawled_pages"].append(extract_page_data(page))
                for sub_page in page.get("sub_pages", []):
                    process_pages(sub_page)
        
        # Process all pages starting from main site
        if research_results["raw_research"]["website_data"]["main_site"]:
            process_pages(research_results["raw_research"]["website_data"]["main_site"])
        
        # Save debug data
        with open(debug_filename, "w", encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2)
        print(f"Debug crawl data saved to {debug_filename}")
        
        # Save summary to markdown file
        with open(summary_filename, "w", encoding='utf-8') as f:
            f.write(research_results["summary"])
        print(f"Research summary saved to {summary_filename}")
        
    except Exception as e:
        print(f"Could not save data files: {e}")
    
    # Return the research results
    return research_results

if __name__ == "__main__":
    asyncio.run(main())