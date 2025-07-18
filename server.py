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
    """Force install browsers with cloud deployment friendly approach"""
    print("🔄 Starting cloud-compatible browser installation...")
    
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
        
        # Simplified installation approaches that work better in cloud environments
        install_commands = [
            # Method 1: Direct python module (most reliable)
            f"{sys.executable} -m playwright install chromium --with-deps",
            f"{sys.executable} -m playwright install chromium",
            
            # Method 2: Using the current python executable
            "python -m playwright install chromium --with-deps",
            "python -m playwright install chromium",
            
            # Method 3: Direct playwright (if available in PATH)
            "playwright install chromium --with-deps",
            "playwright install chromium"
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
                    env=env
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
    """Install browsers in blocking mode during startup with improved error handling"""
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
    
    # Try different installation strategies
    try:
        # Strategy 1: Direct python module approach (most reliable)
        print("🔄 Strategy 1: Using current Python executable")
        success = install_with_python_module()
        
        if success:
            browser_installation_state["installation_in_progress"] = False
            browser_installation_state["is_installed"] = True
            print("🎉 Browser installation completed successfully!")
            return True
        
        # Strategy 2: Force install with comprehensive approach
        print("🔄 Strategy 2: Comprehensive installation approach")
        success = force_install_browsers()
        
        if success:
            browser_installation_state["installation_in_progress"] = False
            browser_installation_state["is_installed"] = True
            print("🎉 Browser installation completed successfully!")
            return True
        
        # Strategy 3: Alternative installation using install_playwright.py
        print("🔄 Strategy 3: Using dedicated installation script")
        success = install_with_script()
        
        if success:
            browser_installation_state["installation_in_progress"] = False
            browser_installation_state["is_installed"] = True
            print("🎉 Browser installation completed successfully!")
            return True
        
    except Exception as e:
        print(f"💥 Error during installation strategies: {e}")
    
    # All strategies failed
    error_msg = "Failed to install Playwright browsers after trying all strategies"
    browser_installation_state["installation_in_progress"] = False
    browser_installation_state["is_installed"] = False
    browser_installation_state["installation_error"] = error_msg
    print(f"❌ {error_msg}")
    return False

def install_with_python_module():
    """Install browsers using the current Python executable"""
    try:
        env = os.environ.copy()
        env['PLAYWRIGHT_BROWSERS_PATH'] = '/tmp/pw-browsers'
        env['PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD'] = '0'
        
        # Ensure directory exists
        os.makedirs("/tmp/pw-browsers", exist_ok=True)
        
        # Use the current Python executable to install
        commands = [
            f"{sys.executable} -m playwright install chromium --with-deps",
            f"{sys.executable} -m playwright install chromium"
        ]
        
        for cmd in commands:
            try:
                print(f"🔄 Trying: {cmd}")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS with: {cmd}")
                    if verify_browser_installation():
                        return True
                else:
                    print(f"❌ FAILED: {cmd}")
                    print(f"   Error: {result.stderr[:200]}...")
                    
            except Exception as e:
                print(f"💥 ERROR: {cmd} - {str(e)}")
        
        return False
        
    except Exception as e:
        print(f"💥 Critical error in python module installation: {e}")
        return False

def install_with_script():
    """Install browsers using the dedicated installation script"""
    try:
        # Try to run the installation script
        result = subprocess.run(
            [sys.executable, "/app/install_playwright.py"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print("✅ Installation script completed successfully")
            return verify_browser_installation()
        else:
            print(f"❌ Installation script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"💥 Error running installation script: {e}")
        return False

# Install browsers BEFORE creating the FastAPI app
print("=" * 60)
print("MCQ SCRAPER - BROWSER INSTALLATION")
print("=" * 60)

# CRITICAL: Install browsers before app starts
try:
    install_success = install_browsers_blocking()
except Exception as e:
    print(f"🚨 CRITICAL ERROR during browser installation: {e}")
    install_success = False

if not install_success:
    print("🚨 CRITICAL: Browser installation failed!")
    print("🚨 App will start but scraping functionality may be limited")
    print("🚨 Consider using a deployment environment with pre-installed browsers")
    
    # Set fallback state
    browser_installation_state["is_installed"] = False
    browser_installation_state["installation_error"] = "Browser installation failed during startup"
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
    Uses intelligent matching including word stems, related terms, and contextual analysis.
    """
    if not question_text or not search_topic:
        return False
    
    # Convert to lowercase for case-insensitive matching
    question_lower = question_text.lower()
    topic_lower = search_topic.lower()
    
    # Enhanced topic matching with word stems and related terms
    topic_variations = [topic_lower]
    
    # Add common word variations and stems
    topic_stems = {
        'biology': ['biological', 'bio', 'organism', 'living', 'life', 'cell', 'plant', 'animal', 'species', 'photosynthesis', 'respiration', 'DNA', 'gene'],
        'physics': ['physical', 'force', 'energy', 'motion', 'matter', 'quantum', 'wave', 'particle', 'newton', 'gravity', 'electricity', 'magnetism'],
        'chemistry': ['chemical', 'reaction', 'compound', 'element', 'molecule', 'atom', 'bond', 'formula', 'water', 'oxygen', 'carbon', 'acid'],
        'heart': ['cardiac', 'cardiovascular', 'circulation', 'blood', 'pulse', 'artery', 'vein', 'pressure', 'organ', 'pump'],
        'mathematics': ['mathematical', 'math', 'equation', 'number', 'calculation', 'formula', 'solve', 'calculate', 'area', 'radius', 'circle', 'triangle'],
        'history': ['historical', 'past', 'ancient', 'period', 'era', 'dynasty', 'empire', 'civilization', 'war', 'battle', 'year', 'century'],
        'geography': ['geographical', 'location', 'place', 'region', 'area', 'continent', 'country', 'climate', 'capital', 'city', 'ocean', 'river', 'mountain'],
        'economics': ['economic', 'economy', 'market', 'trade', 'finance', 'gdp', 'inflation', 'banking', 'money', 'currency', 'investment'],
        'politics': ['political', 'government', 'policy', 'administration', 'governance', 'democracy', 'election', 'president', 'minister', 'parliament'],
        'computer': ['computing', 'software', 'hardware', 'algorithm', 'programming', 'digital', 'binary', 'data', 'internet', 'technology'],
        'science': ['scientific', 'research', 'theory', 'experiment', 'hypothesis', 'discovery', 'planet', 'solar', 'universe', 'nature'],
        'english': ['grammar', 'vocabulary', 'literature', 'language', 'sentence', 'word', 'comprehension', 'reading', 'writing'],
        'reasoning': ['logical', 'logic', 'puzzle', 'pattern', 'sequence', 'analogy', 'verbal', 'analytical', 'solve', 'problem']
    }
    
    # Add stems for the search topic
    if topic_lower in topic_stems:
        topic_variations.extend(topic_stems[topic_lower])
    
    # Add individual words from multi-word topics
    topic_words = topic_lower.split()
    for word in topic_words:
        if len(word) > 3:  # Only add meaningful words
            topic_variations.append(word)
    
    # Also add partial matches (word stems)
    if len(topic_lower) > 4:
        # Add root word (remove common suffixes)
        root_word = topic_lower
        suffixes = ['ical', 'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'ogy', 'ics']
        for suffix in suffixes:
            if root_word.endswith(suffix) and len(root_word) > len(suffix) + 2:
                root_word = root_word[:-len(suffix)]
                topic_variations.append(root_word)
                break
    
    # Remove duplicates and sort by length (longer terms first for better matching)
    topic_variations = sorted(list(set(topic_variations)), key=len, reverse=True)
    
    # Check if any variation is present in question body
    is_relevant = False
    matched_term = ""
    
    for variation in topic_variations:
        if len(variation) > 2 and variation in question_lower:
            is_relevant = True
            matched_term = variation
            break
    
    # If no direct match found, try broader contextual matching
    if not is_relevant:
        # Check if question contains educational/academic keywords
        educational_keywords = ['what', 'which', 'when', 'where', 'how', 'why', 'identify', 'calculate', 'find', 'determine', 'solve', 'name']
        has_educational_structure = any(keyword in question_lower for keyword in educational_keywords)
        
        if has_educational_structure:
            # Check for contextual relevance based on question content
            question_words = set(question_lower.split())
            
            # Context-based matching for different subjects
            subject_contexts = {
                'geography': ['capital', 'country', 'city', 'ocean', 'river', 'mountain', 'continent', 'largest', 'smallest', 'located', 'where'],
                'mathematics': ['calculate', 'area', 'radius', 'circle', 'triangle', 'number', 'equation', 'formula', 'solve', 'add', 'subtract'],
                'biology': ['living', 'organism', 'plant', 'animal', 'cell', 'life', 'species', 'photosynthesis', 'respiration', 'organ'],
                'chemistry': ['formula', 'compound', 'element', 'molecule', 'atom', 'reaction', 'chemical', 'acid', 'base', 'water'],
                'physics': ['force', 'energy', 'motion', 'matter', 'gravity', 'electricity', 'magnetism', 'wave', 'particle'],
                'history': ['war', 'battle', 'year', 'century', 'ancient', 'period', 'empire', 'dynasty', 'when', 'ended', 'started'],
                'politics': ['government', 'democracy', 'election', 'president', 'minister', 'parliament', 'governance', 'policy'],
                'science': ['planet', 'solar', 'universe', 'theory', 'experiment', 'discovery', 'research', 'scientific'],
                'heart': ['organ', 'blood', 'pump', 'body', 'circulation', 'cardiac', 'artery', 'vein'],
                'computer': ['software', 'hardware', 'algorithm', 'programming', 'digital', 'data', 'internet', 'technology'],
                'english': ['grammar', 'vocabulary', 'language', 'sentence', 'word', 'reading', 'writing', 'literature'],
                'reasoning': ['logic', 'puzzle', 'pattern', 'sequence', 'analogy', 'solve', 'problem', 'logical']
            }
            
            # Check if question words overlap with subject context
            if topic_lower in subject_contexts:
                context_words = set(subject_contexts[topic_lower])
                overlap = question_words.intersection(context_words)
                if overlap:
                    is_relevant = True
                    matched_term = f"contextual: {', '.join(overlap)}"
    
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
    Optimized for speed with faster page loading and element selection.
    """
    try:
        # Check if browsers are available
        if not browser_installation_state["is_installed"]:
            if browser_installation_state["installation_error"]:
                print(f"⚠️ Browsers not available: {browser_installation_state['installation_error']}")
                print("🔄 Attempting real-time installation...")
                
                # Try to install browsers now
                try:
                    async with async_playwright() as p:
                        # This will trigger browser download if not present
                        browser = await p.chromium.launch(headless=True)
                        await browser.close()
                        
                        # Update state if successful
                        browser_installation_state["is_installed"] = True
                        browser_installation_state["installation_error"] = None
                        print("✅ Real-time browser installation successful!")
                        
                except Exception as e:
                    print(f"❌ Real-time browser installation failed: {e}")
                    raise Exception(f"Playwright browsers not available: {browser_installation_state['installation_error']}")
            else:
                raise Exception("Playwright browsers not available. Please install them with: playwright install chromium")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()
            
            # Navigate to page with faster loading strategy
            await page.goto(url, wait_until='domcontentloaded', timeout=20000)
            
            # Reduced wait time for faster processing
            await page.wait_for_timeout(1000)
            
            # Extract question using correct selectors with faster approach
            question = ""
            
            # Try both selectors concurrently
            question_selectors = ['h1.questionBody.tag-h1', 'div.questionBody']
            question_elements = await asyncio.gather(
                *[page.query_selector(selector) for selector in question_selectors],
                return_exceptions=True
            )
            
            # Use the first successful result
            question_element = None
            for element in question_elements:
                if element and not isinstance(element, Exception):
                    question_element = element
                    break
            
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
            
            # Extract options, answer, and exam source concurrently for speed
            option_elements_task = page.query_selector_all('li.option')
            answer_element_task = page.query_selector('.solution')
            exam_heading_element_task = page.query_selector('div.pyp-heading')
            exam_title_element_task = page.query_selector('div.pyp-title.line-ellipsis')
            
            # Wait for all tasks to complete
            option_elements, answer_element, exam_heading_element, exam_title_element = await asyncio.gather(
                option_elements_task, answer_element_task, exam_heading_element_task, exam_title_element_task
            )
            
            # Process options
            options = []
            if option_elements:
                option_tasks = [option_elem.inner_text() for option_elem in option_elements]
                option_texts = await asyncio.gather(*option_tasks)
                options = [clean_unwanted_text(text.strip()) for text in option_texts if text.strip()]
            
            # Process answer
            answer = ""
            if answer_element:
                answer = await answer_element.inner_text()
                answer = clean_unwanted_text(answer)
            
            # Process exam source information
            exam_source_heading = ""
            exam_source_title = ""
            
            if exam_heading_element:
                exam_source_heading = await exam_heading_element.inner_text()
                exam_source_heading = clean_unwanted_text(exam_source_heading)
            
            if exam_title_element:
                exam_source_title = await exam_title_element.inner_text()
                exam_source_title = clean_unwanted_text(exam_source_title)
            
            await browser.close()
            
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
    """Process text-based MCQ extraction with optimized concurrent processing"""
    # Extract MCQs with filtering
    mcqs = []
    relevant_mcqs = 0
    irrelevant_mcqs = 0
    
    # Process in batches for better performance
    batch_size = 5  # Process 5 URLs concurrently
    total_batches = (len(links) + batch_size - 1) // batch_size
    
    print(f"🚀 Starting optimized processing: {len(links)} links in {total_batches} batches of {batch_size}")
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(links))
        batch_links = links[batch_start:batch_end]
        
        print(f"📦 Processing batch {batch_num + 1}/{total_batches}: links {batch_start + 1}-{batch_end}")
        
        current_progress = f"🔍 Processing batch {batch_num + 1}/{total_batches} - Smart filtering enabled..."
        update_job_progress(job_id, "running", current_progress, 
                          processed_links=batch_start, mcqs_found=len(mcqs))
        
        # Process batch concurrently
        tasks = [scrape_mcq_content(link, topic) for link in batch_links]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            link_index = batch_start + i + 1
            if isinstance(result, Exception):
                print(f"❌ Error processing link {link_index}: {result}")
                irrelevant_mcqs += 1
            elif result:
                mcqs.append(result)
                relevant_mcqs += 1
                print(f"✅ Found relevant MCQ {link_index}/{len(links)} - Topic: '{topic}' found in question! Total: {len(mcqs)}")
            else:
                irrelevant_mcqs += 1
                print(f"⚠️ Skipped irrelevant MCQ {link_index}/{len(links)} - Topic: '{topic}' not in question. Total: {len(mcqs)}")
        
        # Update progress after each batch
        update_job_progress(job_id, "running", 
                          f"✅ Batch {batch_num + 1}/{total_batches} complete - Found {len(mcqs)} relevant MCQs so far", 
                          processed_links=batch_end, mcqs_found=len(mcqs))
        
        # Small delay between batches to be respectful to the server
        if batch_num < total_batches - 1:  # Don't delay after last batch
            await asyncio.sleep(2)
    
    if not mcqs:
        update_job_progress(job_id, "completed", 
                          f"❌ No relevant MCQs found for '{topic}' across {len(links)} links. Please try another topic or check your search terms.", 
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
        "topic_match_success": round((relevant_mcqs / (relevant_mcqs + irrelevant_mcqs)) * 100, 1) if (relevant_mcqs + irrelevant_mcqs) > 0 else 0,
        "generated_at": datetime.now().isoformat(),
        "api_keys_used": api_key_manager.get_remaining_keys(),
        "exam_focus": exam_type
    }
    
    update_job_progress(job_id, "completed", final_message, 
                      total_links=len(links), processed_links=len(links), 
                      mcqs_found=len(mcqs), pdf_url=pdf_url)

async def process_screenshot_extraction(job_id: str, topic: str, exam_type: str, links: List[str]):
    """Process screenshot-based MCQ extraction for image PDFs with optimized concurrent processing"""
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
    
    # Process in batches for better performance
    batch_size = 3  # Process 3 URLs concurrently for screenshots (more memory intensive)
    total_batches = (len(links) + batch_size - 1) // batch_size
    
    print(f"🚀 Starting optimized screenshot processing: {len(links)} links in {total_batches} batches of {batch_size}")
    
    # Initialize Playwright with multiple browser contexts for concurrent processing
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(links))
            batch_links = links[batch_start:batch_end]
            
            print(f"📦 Processing screenshot batch {batch_num + 1}/{total_batches}: links {batch_start + 1}-{batch_end}")
            
            current_progress = f"📸 Processing screenshot batch {batch_num + 1}/{total_batches} - Smart filtering enabled..."
            update_job_progress(job_id, "running", current_progress, 
                              processed_links=batch_start, mcqs_found=len(screenshots_data))
            
            # Create multiple pages for concurrent processing
            pages = []
            for _ in range(len(batch_links)):
                page = await browser.new_page()
                pages.append(page)
            
            # Process batch concurrently
            tasks = [scrape_testbook_page_with_screenshot(pages[i], batch_links[i], topic) 
                    for i in range(len(batch_links))]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close pages after processing
            for page in pages:
                await page.close()
            
            # Process results
            for i, result in enumerate(batch_results):
                link_index = batch_start + i + 1
                if isinstance(result, Exception):
                    print(f"❌ Error processing screenshot {link_index}: {result}")
                    irrelevant_screenshots += 1
                elif result and result.get('is_relevant'):
                    screenshots_data.append(result)
                    relevant_screenshots += 1
                    print(f"✅ Captured relevant screenshot {link_index}/{len(links)} - Topic: '{topic}' found in question! Total: {len(screenshots_data)}")
                else:
                    irrelevant_screenshots += 1
                    print(f"⚠️ Skipped irrelevant screenshot {link_index}/{len(links)} - Topic: '{topic}' not in question. Total: {len(screenshots_data)}")
            
            # Update progress after each batch
            update_job_progress(job_id, "running", 
                              f"✅ Screenshot batch {batch_num + 1}/{total_batches} complete - Found {len(screenshots_data)} relevant screenshots so far", 
                              processed_links=batch_end, mcqs_found=len(screenshots_data))
            
            # Small delay between batches to be respectful to the server
            if batch_num < total_batches - 1:  # Don't delay after last batch
                await asyncio.sleep(2)
        
        await browser.close()
    
    if not screenshots_data:
        update_job_progress(job_id, "completed", 
                          f"❌ No relevant screenshots found for '{topic}' across {len(links)} links. Please try another topic or check your search terms.", 
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
        "topic_match_success": round((relevant_screenshots / (relevant_screenshots + irrelevant_screenshots)) * 100, 1) if (relevant_screenshots + irrelevant_screenshots) > 0 else 0,
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

@app.post("/api/test-filter")
async def test_filter_logic(request: dict):
    """Test the smart filtering logic with sample data"""
    question_text = request.get("question", "")
    topic = request.get("topic", "")
    
    if not question_text or not topic:
        raise HTTPException(status_code=400, detail="Both 'question' and 'topic' are required")
    
    is_relevant = is_mcq_relevant(question_text, topic)
    
    return {
        "question": question_text,
        "topic": topic,
        "is_relevant": is_relevant,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
