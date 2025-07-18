from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import asyncio
import json
from playwright.async_api import async_playwright
import subprocess
import sys
import time
import threading

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import HexColor, black, darkblue, darkgreen, darkred, white, lightgrey
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.platypus.flowables import Flowable
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import re
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Set Playwright browsers path
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/tmp/pw-browsers'

# Browser installation state
browser_installation_state = {
    "is_installed": False,
    "installation_attempted": False,
    "installation_error": None,
    "installation_in_progress": False
}

def force_install_browsers():
    """Force install browsers with comprehensive approach"""
    print("🔄 Starting comprehensive browser installation...")
    
    try:
        # Ensure directory exists
        os.makedirs("/tmp/pw-browsers", exist_ok=True)
        
        # Set environment variables for installation
        env = os.environ.copy()
        env['PLAYWRIGHT_BROWSERS_PATH'] = '/tmp/pw-browsers'
        env['PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD'] = '0'
        
        # Install system dependencies first
        print("📦 Installing system dependencies...")
        system_deps = [
            "apt-get update -y",
            "apt-get install -y curl wget gnupg lsb-release",
            "apt-get install -y libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgtk-3-0 libgbm1 libasound2",
            "apt-get install -y libxss1 libgconf-2-4 libxtst6 libxrandr2 libasound2 libpangocairo-1.0-0 libatk1.0-0 libcairo-gobject2 libgtk-3-0 libgdk-pixbuf2.0-0",
            "apt-get install -y fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 libatspi2.0-0 libdrm2 libgtk-3-0 libnspr4 libnss3 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libgconf-2-4"
        ]
        
        for dep_cmd in system_deps:
            try:
                subprocess.run(dep_cmd, shell=True, capture_output=True, text=True, timeout=120, env=env)
                print(f"   ✅ {dep_cmd.split()[2] if len(dep_cmd.split()) > 2 else dep_cmd}")
            except:
                print(f"   ⚠️ Failed: {dep_cmd}")
        
        # Multiple installation approaches
        install_commands = [
            # Method 1: Direct playwright install
            "playwright install chromium --with-deps",
            "playwright install chromium",
            
            # Method 2: Python module
            "python -m playwright install chromium --with-deps",
            "python -m playwright install chromium",
            "python3 -m playwright install chromium --with-deps",
            "python3 -m playwright install chromium",
            
            # Method 3: Using pip
            "pip install playwright && playwright install chromium",
            "pip3 install playwright && playwright install chromium",
            
            # Method 4: NPX approach
            "npx playwright install chromium --with-deps",
            "npx playwright install chromium",
            
            # Method 5: Direct download approach
            "wget -O /tmp/install_playwright.py https://playwright.dev/python/docs/ci#installation && python /tmp/install_playwright.py"
        ]
        
        for cmd in install_commands:
            try:
                print(f"🔄 Trying: {cmd}")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                    env=env,
                    cwd="/app"
                )
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS with: {cmd}")
                    print(f"   Output: {result.stdout[:200]}...")
                    
                    # Verify installation
                    if verify_browser_installation():
                        print("✅ Browser installation verified!")
                        return True
                    else:
                        print("⚠️ Installation completed but verification failed")
                        continue
                else:
                    print(f"❌ FAILED: {cmd}")
                    print(f"   Error: {result.stderr[:200]}...")
                    
            except subprocess.TimeoutExpired:
                print(f"⏱️ TIMEOUT: {cmd}")
            except Exception as e:
                print(f"💥 ERROR: {cmd} - {str(e)}")
        
        print("❌ All installation methods failed")
        return False
        
    except Exception as e:
        print(f"💥 Critical error in browser installation: {e}")
        return False

def verify_browser_installation():
    """Verify browser installation with multiple checks"""
    try:
        browser_path = "/tmp/pw-browsers"
        
        if not os.path.exists(browser_path):
            print("❌ Browser directory doesn't exist")
            return False
        
        # Check for browser directories
        browser_found = False
        executable_found = False
        
        for item in os.listdir(browser_path):
            item_path = os.path.join(browser_path, item)
            if os.path.isdir(item_path) and ("chromium" in item.lower() or "chrome" in item.lower()):
                browser_found = True
                print(f"✅ Found browser directory: {item}")
                
                # Check for executables
                possible_executables = [
                    os.path.join(item_path, "chrome-linux", "chrome"),
                    os.path.join(item_path, "chrome-linux", "headless_shell"),
                    os.path.join(item_path, "chromium-linux", "chrome"),
                    os.path.join(item_path, "chromium-linux", "headless_shell"),
                    os.path.join(item_path, "chromium"),
                    os.path.join(item_path, "chrome"),
                    os.path.join(item_path, "headless_shell")
                ]
                
                for executable in possible_executables:
                    if os.path.exists(executable):
                        executable_found = True
                        print(f"✅ Found executable: {executable}")
                        # Check if it's executable
                        if os.access(executable, os.X_OK):
                            print(f"✅ Executable is runnable: {executable}")
                            return True
                        else:
                            print(f"⚠️ Executable not runnable: {executable}")
        
        if browser_found and not executable_found:
            print("⚠️ Browser directory found but no executables")
        elif not browser_found:
            print("❌ No browser directories found")
        
        return False
        
    except Exception as e:
        print(f"❌ Error verifying browser installation: {e}")
        return False

def install_browsers_blocking():
    """Install browsers in blocking mode during startup"""
    global browser_installation_state
    
    print("🚀 Starting browser installation check...")
    
    # Check if already installed
    if verify_browser_installation():
        browser_installation_state["is_installed"] = True
        print("✅ Browsers already installed and verified!")
        return True
    
    # Mark installation as in progress
    browser_installation_state["installation_in_progress"] = True
    browser_installation_state["installation_attempted"] = True
    
    print("🔄 Browsers not found. Starting installation...")
    
    # Force install browsers
    success = force_install_browsers()
    
    # Update state
    browser_installation_state["installation_in_progress"] = False
    browser_installation_state["is_installed"] = success
    
    if success:
        print("🎉 Browser installation completed successfully!")
        return True
    else:
        error_msg = "Failed to install Playwright browsers after trying all methods"
        browser_installation_state["installation_error"] = error_msg
        print(f"❌ {error_msg}")
        return False

# Install browsers BEFORE creating the FastAPI app
print("=" * 60)
print("MCQ SCRAPER - BROWSER INSTALLATION")
print("=" * 60)

# CRITICAL: Install browsers before app starts
install_success = install_browsers_blocking()

if not install_success:
    print("🚨 CRITICAL: Browser installation failed!")
    print("🚨 App may not work properly for scraping tasks")
    print("🚨 Manual installation may be required")
else:
    print("✅ Browser installation successful - App ready to serve!")

print("=" * 60)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Pool Manager
class APIKeyManager:
    def __init__(self):
        # Initialize with API key pool from environment
        api_key_pool = os.getenv("API_KEY_POOL", "")
        self.api_keys = [key.strip() for key in api_key_pool.split(",") if key.strip()]
        self.current_key_index = 0
        self.exhausted_keys = set()
        
        if not self.api_keys:
            raise ValueError("No API keys found in environment")
        
        print(f"🔑 Initialized API Key Manager with {len(self.api_keys)} keys")
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        return self.api_keys[self.current_key_index]
    
    def rotate_key(self) -> Optional[str]:
        """Rotate to the next available key"""
        # Mark current key as exhausted
        current_key = self.api_keys[self.current_key_index]
        self.exhausted_keys.add(current_key)
        print(f"⚠️ Key exhausted: {current_key[:20]}...")
        
        # Find next non-exhausted key
        for i in range(len(self.api_keys)):
            key = self.api_keys[i]
            if key not in self.exhausted_keys:
                self.current_key_index = i
                print(f"🔄 Rotated to key: {key[:20]}...")
                return key
        
        # All keys exhausted
        print("❌ All API keys exhausted!")
        return None
    
    def is_quota_error(self, error_message: str) -> bool:
        """Check if error is related to quota exhaustion"""
        quota_indicators = [
            "quota exceeded",
            "quotaExceeded",
            "rateLimitExceeded",
            "userRateLimitExceeded",
            "dailyLimitExceeded",
            "Too Many Requests"
        ]
        return any(indicator.lower() in error_message.lower() for indicator in quota_indicators)
    
    def get_remaining_keys(self) -> int:
        """Get number of remaining keys"""
        return len(self.api_keys) - len(self.exhausted_keys)

# Initialize API Key Manager
api_key_manager = APIKeyManager()

# Search Engine ID
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "2701a7d64a00d47fd")

# In-memory storage for job progress
job_progress = {}
generated_pdfs = {}

class SearchRequest(BaseModel):
    topic: str
    exam_type: str = "SSC"  # SSC or BPSC
    pdf_format: str = "text"  # text or image

class MCQData(BaseModel):
    question: str
    options: List[str]
    answer: str
    exam_source_heading: str = ""  # "This question was previously asked in"
    exam_source_title: str = ""    # "SSC 2016 Combined Competitive Exam Official paper"
    is_relevant: bool = True       # Indicates if MCQ passed relevance filter

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: str
    total_links: Optional[int] = 0
    processed_links: Optional[int] = 0
    mcqs_found: Optional[int] = 0
    pdf_url: Optional[str] = None

# Custom Border Flowable for PDF
class BorderFlowable(Flowable):
    def __init__(self, width, height, color=HexColor('#2563eb')):
        self.width = width
        self.height = height
        self.color = color
    
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(2)
        self.canv.rect(0, 0, self.width, self.height)

def update_job_progress(job_id: str, status: str, progress: str, **kwargs):
    """Update job progress in memory"""
    if job_id not in job_progress:
        job_progress[job_id] = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "total_links": 0,
            "processed_links": 0,
            "mcqs_found": 0,
            "pdf_url": None
        }
    
    job_progress[job_id].update({
        "status": status,
        "progress": progress,
        **kwargs
    })

def clean_unwanted_text(text: str) -> str:
    """Remove unwanted text strings from scraped content"""
    unwanted_strings = [
        "Download Solution PDF",
        "Download PDF", 
        "Attempt Online",
        "View all BPSC Exam Papers >",
        "View all SSC Exam Papers >",
        "View all BPSC Exam Papers",
        "View all SSC Exam Papers"
    ]
    
    cleaned_text = text
    for unwanted in unwanted_strings:
        cleaned_text = cleaned_text.replace(unwanted, "")
    
    # Remove extra whitespace and newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF generation by removing unwanted characters and formatting"""
    if not text:
        return ""
    
    # Remove unwanted characters and normalize whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any remaining unwanted strings
    unwanted_patterns = [
        r'Download\s+Solution\s+PDF',
        r'Download\s+PDF',
        r'Attempt\s+Online',
        r'View\s+all\s+\w+\s+Exam\s+Papers\s*>?'
    ]
    
    for pattern in unwanted_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

async def capture_page_screenshot(page, url: str, topic: str) -> Optional[bytes]:
    """Capture screenshot of complete MCQ content (question + options + solution)"""
    try:
        print(f"📸 Capturing screenshot for URL: {url}")
        
        # Navigate to the page
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)  # Wait for page to fully load
        
        # Set viewport to ensure good quality
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.wait_for_timeout(1000)  # Wait for viewport adjustment
        
        # Find all MCQ content elements
        mcq_elements = []
        
        # Find question element
        question_element = await page.query_selector('h1.questionBody.tag-h1')
        if not question_element:
            question_element = await page.query_selector('div.questionBody')
        if question_element:
            mcq_elements.append(question_element)
            print(f"📝 Found question element")
        
        # Find option elements
        option_elements = await page.query_selector_all('li.option')
        if option_elements:
            mcq_elements.extend(option_elements)
            print(f"📝 Found {len(option_elements)} option elements")
        
        # Find solution element
        solution_element = await page.query_selector('.solution')
        if solution_element:
            mcq_elements.append(solution_element)
            print(f"📝 Found solution element")
        
        # Find exam source elements
        exam_heading_element = await page.query_selector('div.pyp-heading')
        if exam_heading_element:
            mcq_elements.append(exam_heading_element)
            print(f"📝 Found exam heading element")
        
        exam_title_element = await page.query_selector('div.pyp-title.line-ellipsis')
        if exam_title_element:
            mcq_elements.append(exam_title_element)
            print(f"📝 Found exam title element")
        
        if not mcq_elements:
            print(f"❌ No MCQ elements found on {url}")
            return None
        
        # Calculate bounding box for all MCQ elements
        bounding_boxes = []
        for element in mcq_elements:
            try:
                box = await element.bounding_box()
                if box:
                    bounding_boxes.append(box)
            except Exception as e:
                print(f"⚠️ Could not get bounding box for element: {e}")
                continue
        
        if not bounding_boxes:
            print(f"❌ Could not get bounding boxes for MCQ elements on {url}")
            return None
        
        # Calculate combined bounding box
        min_x = min(box['x'] for box in bounding_boxes)
        min_y = min(box['y'] for box in bounding_boxes)
        max_x = max(box['x'] + box['width'] for box in bounding_boxes)
        max_y = max(box['y'] + box['height'] for box in bounding_boxes)
        
        # Add some padding around the content
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        
        # Get viewport dimensions to ensure we don't go beyond page boundaries
        viewport = await page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
        screenshot_width = min(max_x - min_x + padding * 2, viewport['width'] - min_x)
        screenshot_height = min(max_y - min_y + padding * 2, viewport['height'] - min_y)
        
        print(f"📐 Screenshot dimensions: {screenshot_width}x{screenshot_height} at ({min_x}, {min_y})")
        
        # Capture screenshot of the MCQ content area
        screenshot = await page.screenshot(
            clip={
                "x": min_x,
                "y": min_y,
                "width": screenshot_width,
                "height": screenshot_height
            },
            type="png"
        )
        
        print(f"✅ Screenshot captured successfully for {url} - MCQ content area")
        return screenshot
        
    except Exception as e:
        print(f"❌ Error capturing screenshot for {url}: {str(e)}")
        return None

async def scrape_testbook_page_with_screenshot(page, url: str, topic: str) -> Optional[dict]:
    """Scrape Testbook page and capture screenshot - modified for image PDFs"""
    try:
        print(f"🔍 Processing URL with screenshot: {url}")
        
        # Navigate to the page
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)
        
        # Check if page has MCQ content using the SAME selectors as text scraping
        # Try h1.questionBody.tag-h1 first
        question_element = await page.query_selector('h1.questionBody.tag-h1')
        if not question_element:
            # Fallback to .questionBody as div
            question_element = await page.query_selector('div.questionBody')
        
        if not question_element:
            print(f"❌ No MCQ content found on {url} - no questionBody element")
            return None
        
        # Extract basic MCQ data for filtering
        mcq_data = await scrape_mcq_content(url, topic)
        
        if not mcq_data or not mcq_data.is_relevant:
            print(f"❌ MCQ not relevant for topic '{topic}' on {url}")
            return None
        
        # Capture screenshot of top 50%
        screenshot = await capture_page_screenshot(page, url, topic)
        
        if not screenshot:
            print(f"❌ Failed to capture screenshot for {url}")
            return None
        
        return {
            "url": url,
            "screenshot": screenshot,
            "mcq_data": mcq_data,
            "is_relevant": mcq_data.is_relevant
        }
        
    except Exception as e:
        print(f"❌ Error processing {url} with screenshot: {str(e)}")
        return None

def is_mcq_relevant(question_text: str, search_topic: str) -> bool:
    """
    Check if MCQ is relevant by verifying if the search topic is present in the question body.
    IMPORTANT: Only check questionBody, NOT options or solution.
    Uses intelligent matching including word stems and related terms.
    """
    if not question_text or not search_topic:
        print(f"🔍 DEBUG: Empty question_text ({len(question_text) if question_text else 0}) or search_topic ({len(search_topic) if search_topic else 0})")
        return False
    
    # Convert to lowercase for case-insensitive matching
    question_lower = question_text.lower()
    topic_lower = search_topic.lower()
    
    # Enhanced topic matching with word stems and related terms
    topic_variations = [topic_lower]
    
    # Add common word variations and stems
    topic_stems = {
        'biology': ['biological', 'bio', 'organism', 'living', 'life'],
        'physics': ['physical', 'force', 'energy', 'motion', 'matter'],
        'chemistry': ['chemical', 'reaction', 'compound', 'element', 'molecule'],
        'heart': ['cardiac', 'cardiovascular', 'circulation', 'blood', 'pulse'],
        'mathematics': ['mathematical', 'math', 'equation', 'number', 'calculation'],
        'history': ['historical', 'past', 'ancient', 'period', 'era'],
        'geography': ['geographical', 'location', 'place', 'region', 'area'],
        'economics': ['economic', 'economy', 'market', 'trade', 'finance'],
        'politics': ['political', 'government', 'policy', 'administration', 'governance']
    }
    
    # Add stems for the search topic
    if topic_lower in topic_stems:
        topic_variations.extend(topic_stems[topic_lower])
    
    # Also add partial matches (word stems)
    if len(topic_lower) > 4:
        # Add root word (remove common suffixes)
        root_word = topic_lower
        suffixes = ['ical', 'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
        for suffix in suffixes:
            if root_word.endswith(suffix) and len(root_word) > len(suffix) + 2:
                root_word = root_word[:-len(suffix)]
                topic_variations.append(root_word)
                break
    
    # Check if any variation is present in question body
    is_relevant = False
    matched_term = ""
    
    for variation in topic_variations:
        if variation in question_lower:
            is_relevant = True
            matched_term = variation
            break
    
    # Enhanced debug logging
    print(f"🔍 DEBUG: Question (first 100 chars): '{question_lower[:100]}...'")
    print(f"🔍 DEBUG: Topic to find: '{topic_lower}'")
    print(f"🔍 DEBUG: Topic variations: {topic_variations}")
    print(f"🔍 DEBUG: Matched term: '{matched_term}' - Topic found in question: {is_relevant}")
    
    return is_relevant

async def search_google_custom(topic: str, exam_type: str = "SSC") -> List[str]:
    """Search Google Custom Search API with automatic key rotation - SSC/BPSC FOCUSED"""
    # NEW SEARCH QUERY FORMAT - SSC/BPSC FOCUSED
    if exam_type.upper() == "BPSC":
        # Use broader search terms for BPSC to find more relevant content
        query = f'{topic} Testbook [Solved] "This question was previously asked in" ("BPSC" OR "Bihar Public Service Commission" OR "BPSC Combined" OR "BPSC Prelims") '
    else:  # Default to SSC
        query = f'{topic} Testbook [Solved] "This question was previously asked in" "SSC" '
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    headers = {
        "Referer": "https://testbook.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    all_testbook_links = []
    start_index = 1
    max_results = 100
    
    try:
        while start_index <= max_results:
            current_key = api_key_manager.get_current_key()
            
            params = {
                "key": current_key,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "num": 10,
                "start": start_index
            }
            
            print(f"🔍 Fetching results {start_index}-{start_index+9} for topic: {topic}")
            print(f"🔑 Using key: {current_key[:20]}... (Remaining: {api_key_manager.get_remaining_keys()})")
            
            response = requests.get(base_url, params=params, headers=headers)
            
            # Check for quota errors
            if response.status_code == 429 or (response.status_code == 403 and "quota" in response.text.lower()):
                print(f"⚠️ Quota exceeded for current key. Attempting rotation...")
                
                # Try to rotate key
                next_key = api_key_manager.rotate_key()
                if next_key is None:
                    print("❌ All API keys exhausted!")
                    raise Exception("All Servers are exhausted due to intense use")
                
                # Retry with new key
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if "items" not in data or len(data["items"]) == 0:
                print(f"No more results found after {start_index-1} results")
                break
            
            # Extract Testbook links from this batch
            batch_links = []
            for item in data["items"]:
                link = item.get("link", "")
                if "testbook.com" in link:
                    batch_links.append(link)
            
            all_testbook_links.extend(batch_links)
            print(f"✅ Found {len(batch_links)} Testbook links in this batch. Total so far: {len(all_testbook_links)}")
            
            # Check if we got fewer than 10 results (last page)
            if len(data["items"]) < 10:
                print(f"Reached end of results with {len(data['items'])} items in last batch")
                break
            
            start_index += 10
            
            # Small delay to be respectful to the API
            await asyncio.sleep(0.5)
        
        print(f"✅ Total Testbook links found: {len(all_testbook_links)}")
        return all_testbook_links
        
    except Exception as e:
        print(f"❌ Error searching Google: {e}")
        if "All Servers are exhausted due to intense use" in str(e):
            raise e
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return []

async def scrape_mcq_content(url: str, search_topic: str) -> Optional[MCQData]:
    """
    Scrape MCQ content with topic relevance filtering and exam source extraction.
    CRITICAL: Only scrape MCQs where questionBody contains the search topic.
    """
    try:
        # Check if browsers are available
        if not browser_installation_state["is_installed"]:
            if browser_installation_state["installation_error"]:
                raise Exception(f"Playwright browsers not available: {browser_installation_state['installation_error']}")
            else:
                raise Exception("Playwright browsers not available. Please install them with: playwright install chromium")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()
            
            # Navigate to page
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for content to load
            await page.wait_for_timeout(2000)
            
            # Extract question using correct selectors
            question = ""
            
            # Try h1.questionBody.tag-h1 first
            question_element = await page.query_selector('h1.questionBody.tag-h1')
            if question_element:
                question = await question_element.inner_text()
            else:
                # Fallback to .questionBody as div
                question_element = await page.query_selector('div.questionBody')
                if question_element:
                    question = await question_element.inner_text()
            
            # Clean question text
            if question:
                question = clean_unwanted_text(question)
            
            # CRITICAL NEW FILTERING: Check topic relevance BEFORE processing options/solution
            print(f"🔍 DEBUG: Extracted question text: '{question[:100]}...' (length: {len(question)})")
            print(f"🔍 DEBUG: Search topic: '{search_topic}'")
            
            if not is_mcq_relevant(question, search_topic):
                print(f"❌ MCQ skipped - topic '{search_topic}' not found in question body")
                print(f"🔍 DEBUG: Question text was: '{question}'")
                await browser.close()
                return None
            
            print(f"✅ MCQ relevant - topic '{search_topic}' found in question body")
            
            # Extract options using correct selector
            options = []
            option_elements = await page.query_selector_all('li.option')
            for option_elem in option_elements:
                option_text = await option_elem.inner_text()
                if option_text.strip():
                    options.append(clean_unwanted_text(option_text.strip()))
            
            # Extract answer and solution
            answer = ""
            answer_element = await page.query_selector('.solution')
            if answer_element:
                answer = await answer_element.inner_text()
            
            # NEW: Extract exam source information
            exam_source_heading = ""
            exam_source_title = ""
            
            # Extract exam source heading
            exam_heading_element = await page.query_selector('div.pyp-heading')
            if exam_heading_element:
                exam_source_heading = await exam_heading_element.inner_text()
                exam_source_heading = clean_unwanted_text(exam_source_heading)
            
            # Extract exam source title
            exam_title_element = await page.query_selector('div.pyp-title.line-ellipsis')
            if exam_title_element:
                exam_source_title = await exam_title_element.inner_text()
                exam_source_title = clean_unwanted_text(exam_source_title)
            
            await browser.close()
            
            # Clean unwanted text from answer
            if answer:
                answer = clean_unwanted_text(answer)
            
            # Return enhanced MCQ data if we found content
            if question and (options or answer):
                return MCQData(
                    question=question.strip(),
                    options=options,
                    answer=answer.strip(),
                    exam_source_heading=exam_source_heading.strip(),
                    exam_source_title=exam_source_title.strip(),
                    is_relevant=True
                )
            
            return None
            
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def generate_pdf(mcqs: List[MCQData], topic: str, job_id: str, relevant_mcqs: int, irrelevant_mcqs: int, total_links: int) -> str:
    """Generate a professionally formatted PDF with enhanced visual design and filtering statistics"""
    try:
        # Create PDFs directory if it doesn't exist
        pdf_dir = Path("/app/pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        filename = f"Testbook_MCQs_{topic.replace(' ', '_')}_{job_id}.pdf"
        filepath = pdf_dir / filename
        
        # Create PDF document with optimized settings
        doc = SimpleDocTemplate(str(filepath), pagesize=A4, 
                              topMargin=0.6*inch, bottomMargin=0.6*inch,
                              leftMargin=0.6*inch, rightMargin=0.6*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Define professional color scheme
        primary_color = HexColor('#2563eb')      # Professional blue
        secondary_color = HexColor('#1e40af')    # Darker blue
        accent_color = HexColor('#10b981')       # Success green
        text_color = HexColor('#1f2937')         # Dark gray
        light_color = HexColor('#f3f4f6')        # Light gray
        
        # Enhanced custom styles with professional colors
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=25,
            alignment=TA_CENTER,
            textColor=primary_color,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=primary_color,
            borderPadding=10
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=secondary_color,
            fontName='Helvetica-Bold'
        )
        
        stats_style = ParagraphStyle(
            'StatsStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=accent_color,
            fontName='Helvetica-Bold'
        )
        
        question_header_style = ParagraphStyle(
            'QuestionHeaderStyle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=10,
            fontName='Helvetica-Bold',
            textColor=primary_color,
            borderWidth=1,
            borderColor=primary_color,
            borderPadding=8,
            backColor=light_color
        )
        
        question_style = ParagraphStyle(
            'QuestionStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leftIndent=10,
            rightIndent=10,
            fontName='Helvetica-Bold',
            textColor=text_color,
            leading=18  # Improved line spacing
        )
        
        option_style = ParagraphStyle(
            'OptionStyle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=25,
            rightIndent=10,
            textColor=text_color,
            leading=16
        )
        
        answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=15,
            leftIndent=10,
            rightIndent=10,
            textColor=accent_color,
            fontName='Helvetica-Bold',
            leading=16
        )
        
        exam_source_style = ParagraphStyle(
            'ExamSourceStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            leftIndent=10,
            rightIndent=10,
            textColor=secondary_color,
            fontName='Helvetica-Oblique',
            leading=14
        )
        
        # Build PDF content
        story = []
        
        # Professional header with border
        story.append(Paragraph("📚 COMPREHENSIVE SSC MCQ COLLECTION", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Subject: <b>{topic.upper()}</b>", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Enhanced statistics section with professional table INCLUDING FILTERING STATS
        stats_data = [
            ['📊 Collection Statistics', ''],
            ['Search Topic', f'{topic}'],
            ['Total Relevant Questions', f'{len(mcqs)}'],  # Only relevant MCQs
            ['Filtering Applied', 'Topic-based (Question Body Only)'],
            ['Generated On', f'{datetime.now().strftime("%B %d, %Y at %I:%M %p")}'],
            ['Source', 'Testbook.com (SSC Focus)'],
            ['Quality', 'Filtered & Professional Grade']
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), light_color),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 0.2*inch))
        
        # NEW: Filtering Statistics Table
        filtering_data = [
            ['🔍 Smart Filtering Results', ''],
            ['Total Links Searched', f'{total_links}'],
            ['Relevant MCQs Found', f'{relevant_mcqs}'],
            ['Irrelevant MCQs Skipped', f'{irrelevant_mcqs}'],
            ['Filtering Efficiency', f'{round((relevant_mcqs / total_links) * 100, 1)}%'],
            ['Topic Match Success', f'{round((relevant_mcqs / (relevant_mcqs + irrelevant_mcqs)) * 100, 1)}%']
        ]
        
        filtering_table = Table(filtering_data, colWidths=[2.5*inch, 2.5*inch])
        filtering_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), accent_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), light_color),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, accent_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(filtering_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Professional separator
        story.append(Paragraph("═" * 80, ParagraphStyle('separator', textColor=primary_color, alignment=TA_CENTER)))
        story.append(PageBreak())
        
        # Table of Contents for large collections
        if len(mcqs) > 15:
            story.append(Paragraph("📋 TABLE OF CONTENTS", question_header_style))
            story.append(Spacer(1, 0.2*inch))
            
            toc_data = [['No.', 'Question Preview', 'Page']]
            for i, mcq in enumerate(mcqs[:50], 1):  # Limit TOC to first 50 questions
                question_preview = (mcq.question[:70] + "...") if len(mcq.question) > 70 else mcq.question
                page_num = f"Page {((i-1)//2) + 3}"  # Rough page calculation
                toc_data.append([str(i), question_preview, page_num])
            
            toc_table = Table(toc_data, colWidths=[0.5*inch, 4*inch, 1*inch])
            toc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, primary_color),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
            ]))
            
            story.append(toc_table)
            story.append(PageBreak())
        
        # Enhanced MCQ content with professional formatting
        for i, mcq in enumerate(mcqs, 1):
            # Professional question header
            story.append(Paragraph(f"QUESTION {i} OF {len(mcqs)}", question_header_style))
            story.append(Spacer(1, 0.1*inch))
            
            # NEW: Exam source information display
            if mcq.exam_source_heading or mcq.exam_source_title:
                exam_source_text = ""
                if mcq.exam_source_heading:
                    exam_source_text += f"📋 {mcq.exam_source_heading}"
                if mcq.exam_source_title:
                    exam_source_text += f" - {mcq.exam_source_title}"
                
                if exam_source_text:
                    story.append(Paragraph(exam_source_text, exam_source_style))
                    story.append(Spacer(1, 0.1*inch))
            
            # Question content with enhanced formatting
            question_text = mcq.question.replace('\n', '<br/>')
            story.append(Paragraph(f"<b>Q{i}:</b> {question_text}", question_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Options with professional styling
            if mcq.options:
                story.append(Paragraph("📝 <b>OPTIONS:</b>", option_style))
                for j, option in enumerate(mcq.options):
                    option_letter = chr(ord('A') + j) if j < 26 else f"Option {j+1}"
                    option_text = option.replace('\n', '<br/>')
                    story.append(Paragraph(f"<b>{option_letter}.</b> {option_text}", option_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Answer with professional formatting
            if mcq.answer:
                story.append(Paragraph("💡 <b>ANSWER & DETAILED SOLUTION:</b>", answer_style))
                answer_text = mcq.answer.replace('\n', '<br/>')
                story.append(Paragraph(answer_text, answer_style))
            
            # Professional separator between questions
            story.append(Spacer(1, 0.25*inch))
            story.append(Paragraph("─" * 100, ParagraphStyle('divider', textColor=primary_color, alignment=TA_CENTER, fontSize=8)))
            story.append(Spacer(1, 0.25*inch))
            
            # Add page break every 2 questions for better readability
            if i % 2 == 0 and i < len(mcqs):
                story.append(PageBreak())
        
        # Professional footer section
        story.append(PageBreak())
        story.append(Paragraph("🎯 COLLECTION COMPLETE", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary table with filtering info
        summary_data = [
            ['📈 SUMMARY STATISTICS', ''],
            ['Total Questions Collected', f'{len(mcqs)}'],
            ['Subject Area', f'{topic}'],
            ['Source Platform', 'Testbook.com (SSC Focus)'],
            ['Quality Level', 'Professional Grade'],
            ['Filtering Applied', 'Smart Topic-based Filtering'],
            ['Generated By', 'Testbook MCQ Extractor'],
            ['Generated On', f'{datetime.now().strftime("%B %d, %Y at %I:%M %p")}']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), accent_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), light_color),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, accent_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Thank you for using the Enhanced SSC-Focused Testbook MCQ Extractor!", 
                             ParagraphStyle('thanks', textColor=primary_color, alignment=TA_CENTER, fontSize=12, fontName='Helvetica-Bold')))
        
        # Build PDF with enhanced settings
        doc.build(story)
        
        print(f"✅ Professional PDF generated successfully: {filename} with {len(mcqs)} relevant MCQs")
        return filename
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        raise

def generate_image_based_pdf(screenshots_data: List[dict], topic: str, exam_type: str = "SSC") -> str:
    """Generate PDF with screenshots of MCQ pages"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.colors import HexColor
        import io
        from PIL import Image as PILImage
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mcq_screenshots_{topic}_{exam_type}_{timestamp}.pdf"
        filepath = f"/app/pdfs/{filename}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2563eb'),
            alignment=TA_CENTER
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=HexColor('#1e40af'),
            alignment=TA_CENTER
        )
        
        # Title page
        story.append(Paragraph(f"📚 {exam_type} MCQ COLLECTION", title_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Subject: <b>{topic.upper()}</b>", header_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Format: Image Screenshots", header_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Total Questions: {len(screenshots_data)}", header_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", header_style))
        story.append(PageBreak())
        
        # Add screenshots
        for i, screenshot_item in enumerate(screenshots_data, 1):
            # Add page header
            story.append(Paragraph(f"Question {i}", header_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Add URL reference
            url_style = ParagraphStyle(
                'URL',
                parent=styles['Normal'],
                fontSize=10,
                textColor=HexColor('#6b7280'),
                alignment=TA_LEFT
            )
            story.append(Paragraph(f"Source: {screenshot_item['url']}", url_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Convert screenshot bytes to PIL Image
            screenshot_pil = PILImage.open(io.BytesIO(screenshot_item['screenshot']))
            
            # Create reportlab Image from PIL Image
            img_buffer = io.BytesIO()
            screenshot_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Calculate dimensions to fit page
            page_width = letter[0] - 2*inch  # Leave margins
            page_height = letter[1] - 3*inch  # Leave space for header/footer
            
            # Calculate aspect ratio
            img_width, img_height = screenshot_pil.size
            aspect_ratio = img_width / img_height
            
            # Calculate display size
            if aspect_ratio > 1:  # Landscape
                display_width = min(page_width, 6*inch)
                display_height = display_width / aspect_ratio
            else:  # Portrait
                display_height = min(page_height, 8*inch)
                display_width = display_height * aspect_ratio
            
            # Add image to story
            img = Image(img_buffer, width=display_width, height=display_height)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
            
            # Add page break except for last item
            if i < len(screenshots_data):
                story.append(PageBreak())
        
        # Summary page
        story.append(PageBreak())
        story.append(Paragraph("📊 SUMMARY", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        summary_text = f"""
        Total Screenshots: {len(screenshots_data)}<br/>
        Topic: {topic}<br/>
        Exam Type: {exam_type}<br/>
        Format: Image-based PDF<br/>
        Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        Source: Testbook.com
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Image-based PDF generated successfully: {filename} with {len(screenshots_data)} screenshots")
        return filename
        
    except Exception as e:
        print(f"❌ Error generating image-based PDF: {e}")
        raise

async def process_mcq_extraction(job_id: str, topic: str, exam_type: str = "SSC", pdf_format: str = "text"):
    """
    Enhanced processing with topic-based filtering and support for both text and image PDFs.
    CRITICAL: Only process MCQs where questionBody contains the search topic.
    """
    try:
        update_job_progress(job_id, "running", f"🔍 Searching for {exam_type} '{topic}' results with smart filtering...")
        
        # Search for ALL available links with key rotation
        links = await search_google_custom(topic, exam_type)
        
        if not links:
            update_job_progress(job_id, "completed", f"❌ No {exam_type} results found for '{topic}'. Please try another topic.", 
                              total_links=0, processed_links=0, mcqs_found=0)
            return
        
        update_job_progress(job_id, "running", f"✅ Found {len(links)} {exam_type} links. Starting smart filtering extraction...", 
                          total_links=len(links))
        
        if pdf_format == "image":
            # Process screenshots for image-based PDF
            await process_screenshot_extraction(job_id, topic, exam_type, links)
        else:
            # Process text-based PDF (existing functionality)
            await process_text_extraction(job_id, topic, exam_type, links)
        
    except Exception as e:
        error_message = str(e)
        if "All Servers are exhausted due to intense use" in error_message:
            update_job_progress(job_id, "error", "❌ All Servers are exhausted due to intense use")
        else:
            update_job_progress(job_id, "error", f"❌ Error: {error_message}")
        print(f"❌ Error in process_mcq_extraction: {e}")

async def process_text_extraction(job_id: str, topic: str, exam_type: str, links: List[str]):
    """Process text-based MCQ extraction"""
    # Extract MCQs with filtering
    mcqs = []
    relevant_mcqs = 0
    irrelevant_mcqs = 0
    
    for i, link in enumerate(links, 1):
        current_progress = f"🔍 Processing link {i} of {len(links)} - Smart filtering enabled..."
        update_job_progress(job_id, "running", current_progress, 
                          processed_links=i-1, mcqs_found=len(mcqs))
        
        # Pass search topic to scraping function for filtering
        mcq_data = await scrape_mcq_content(link, topic)
        if mcq_data:
            mcqs.append(mcq_data)
            relevant_mcqs += 1
            update_job_progress(job_id, "running", 
                              f"✅ Found relevant MCQ {i}/{len(links)} - Topic: '{topic}' found in question! Total: {len(mcqs)}", 
                              processed_links=i, mcqs_found=len(mcqs))
        else:
            irrelevant_mcqs += 1
            update_job_progress(job_id, "running", 
                              f"⚠️ Skipped irrelevant MCQ {i}/{len(links)} - Topic: '{topic}' not in question. Total: {len(mcqs)}", 
                              processed_links=i, mcqs_found=len(mcqs))
        
        # Small delay between scrapes to be respectful
        await asyncio.sleep(1)
    
    if not mcqs:
        update_job_progress(job_id, "completed", 
                          f"❌ No relevant MCQs found for '{topic}' across {len(links)} links. Please try another topic.", 
                          total_links=len(links), processed_links=len(links), mcqs_found=0)
        return
    
    # Enhanced completion message with filtering statistics
    final_message = f"✅ Smart filtering complete! Found {relevant_mcqs} relevant MCQs (topic '{topic}' in question body) from {len(links)} total links. Skipped {irrelevant_mcqs} irrelevant MCQs."
    
    # Generate professional PDF with filtering statistics
    update_job_progress(job_id, "running", 
                      f"📄 Generating professional PDF with {len(mcqs)} relevant MCQs and filtering statistics...", 
                      total_links=len(links), processed_links=len(links), mcqs_found=len(mcqs))
    
    pdf_filename = generate_pdf(mcqs, topic, job_id, relevant_mcqs, irrelevant_mcqs, len(links))
    pdf_url = f"/api/download/{pdf_filename}"
    
    # Store PDF info with enhanced metadata
    generated_pdfs[pdf_filename] = {
        "filename": pdf_filename,
        "topic": topic,
        "mcq_count": len(mcqs),
        "total_links_searched": len(links),
        "relevant_mcqs": relevant_mcqs,
        "irrelevant_mcqs": irrelevant_mcqs,
        "filtering_efficiency": round((relevant_mcqs / len(links)) * 100, 1),
        "topic_match_success": round((relevant_mcqs / (relevant_mcqs + irrelevant_mcqs)) * 100, 1),
        "generated_at": datetime.now().isoformat(),
        "api_keys_used": api_key_manager.get_remaining_keys(),
        "exam_focus": exam_type
    }
    
    update_job_progress(job_id, "completed", final_message, 
                      total_links=len(links), processed_links=len(links), 
                      mcqs_found=len(mcqs), pdf_url=pdf_url)

async def process_screenshot_extraction(job_id: str, topic: str, exam_type: str, links: List[str]):
    """Process screenshot-based MCQ extraction for image PDFs"""
    screenshots_data = []
    relevant_screenshots = 0
    irrelevant_screenshots = 0
    
    # Check if browsers are available
    if not browser_installation_state["is_installed"]:
        if browser_installation_state["installation_error"]:
            update_job_progress(job_id, "error", f"❌ Playwright browsers not available: {browser_installation_state['installation_error']}")
        else:
            update_job_progress(job_id, "error", "❌ Playwright browsers not available. Please install them with: playwright install chromium")
        return
    
    # Initialize Playwright for screenshot capture
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for i, link in enumerate(links, 1):
            current_progress = f"📸 Capturing screenshot {i} of {len(links)} - Smart filtering enabled..."
            update_job_progress(job_id, "running", current_progress, 
                              processed_links=i-1, mcqs_found=len(screenshots_data))
            
            # Capture screenshot with filtering
            screenshot_data = await scrape_testbook_page_with_screenshot(page, link, topic)
            
            if screenshot_data and screenshot_data['is_relevant']:
                screenshots_data.append(screenshot_data)
                relevant_screenshots += 1
                update_job_progress(job_id, "running", 
                                  f"✅ Captured relevant screenshot {i}/{len(links)} - Topic: '{topic}' found in question! Total: {len(screenshots_data)}", 
                                  processed_links=i, mcqs_found=len(screenshots_data))
            else:
                irrelevant_screenshots += 1
                update_job_progress(job_id, "running", 
                                  f"⚠️ Skipped irrelevant screenshot {i}/{len(links)} - Topic: '{topic}' not in question. Total: {len(screenshots_data)}", 
                                  processed_links=i, mcqs_found=len(screenshots_data))
            
            # Small delay between scrapes to be respectful
            await asyncio.sleep(1)
        
        await browser.close()
    
    if not screenshots_data:
        update_job_progress(job_id, "completed", 
                          f"❌ No relevant screenshots found for '{topic}' across {len(links)} links. Please try another topic.", 
                          total_links=len(links), processed_links=len(links), mcqs_found=0)
        return
    
    # Enhanced completion message with filtering statistics
    final_message = f"✅ Screenshot filtering complete! Found {relevant_screenshots} relevant screenshots (topic '{topic}' in question body) from {len(links)} total links. Skipped {irrelevant_screenshots} irrelevant screenshots."
    
    # Generate image-based PDF with screenshots
    update_job_progress(job_id, "running", 
                      f"📄 Generating image-based PDF with {len(screenshots_data)} relevant screenshots...", 
                      total_links=len(links), processed_links=len(links), mcqs_found=len(screenshots_data))
    
    pdf_filename = generate_image_based_pdf(screenshots_data, topic, exam_type)
    pdf_url = f"/api/download/{pdf_filename}"
    
    # Store PDF info with enhanced metadata
    generated_pdfs[pdf_filename] = {
        "filename": pdf_filename,
        "topic": topic,
        "mcq_count": len(screenshots_data),
        "total_links_searched": len(links),
        "relevant_mcqs": relevant_screenshots,
        "irrelevant_mcqs": irrelevant_screenshots,
        "filtering_efficiency": round((relevant_screenshots / len(links)) * 100, 1),
        "topic_match_success": round((relevant_screenshots / (relevant_screenshots + irrelevant_screenshots)) * 100, 1),
        "generated_at": datetime.now().isoformat(),
        "api_keys_used": api_key_manager.get_remaining_keys(),
        "exam_focus": exam_type,
        "pdf_format": "image"
    }
    
    update_job_progress(job_id, "completed", final_message, 
                      total_links=len(links), processed_links=len(links), 
                      mcqs_found=len(screenshots_data), pdf_url=pdf_url)

# ============================================
# MAIN API ENDPOINTS
# ============================================

@app.post("/api/generate-mcq-pdf")
async def generate_mcq_pdf(request: SearchRequest, background_tasks: BackgroundTasks):
    """Generate MCQ PDF with enhanced filtering and format support"""
    job_id = str(uuid.uuid4())
    
    # Input validation
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")
    
    if request.exam_type not in ["SSC", "BPSC"]:
        raise HTTPException(status_code=400, detail="Exam type must be SSC or BPSC")
    
    if request.pdf_format not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="PDF format must be text or image")
    
    # Start background task
    background_tasks.add_task(
        process_mcq_extraction, 
        job_id, 
        request.topic.strip(), 
        request.exam_type, 
        request.pdf_format
    )
    
    return JobStatus(
        job_id=job_id,
        status="running",
        progress=f"🔍 Starting {request.exam_type} MCQ extraction for '{request.topic}' ({request.pdf_format} format)...",
        total_links=0,
        processed_links=0,
        mcqs_found=0
    )

@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**job_progress[job_id])

@app.get("/api/download/{filename}")
async def download_pdf(filename: str):
    """Download generated PDF"""
    file_path = Path("/app/pdfs") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename not in generated_pdfs:
        raise HTTPException(status_code=404, detail="PDF not found in records")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/pdf"
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MCQ Scraper API",
        "browsers_installed": browser_installation_state["is_installed"],
        "browsers_error": browser_installation_state["installation_error"],
        "installation_attempted": browser_installation_state["installation_attempted"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
