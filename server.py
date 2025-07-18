from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import asyncio
import json
from playwright.async_api import async_playwright, Browser, BrowserContext
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

# ENHANCED Browser Pool Manager - ROBUST CLOUD DEPLOYMENT VERSION
class RobustBrowserPoolManager:
    """
    Enhanced browser manager with robust error handling for cloud deployments
    Handles browser crashes, resource constraints, and connection failures
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright_instance = None
        self.is_initialized = False
        self.lock = asyncio.Lock()
        self.retry_count = 0
        self.max_retries = 3
        self.last_error = None
        
    async def initialize(self):
        """Initialize browser with enhanced cloud-friendly settings"""
        async with self.lock:
            if self.is_initialized and self.browser and not self.browser.is_connected():
                print("⚠️ Browser disconnected, reinitializing...")
                await self._cleanup()
                self.is_initialized = False
            
            if self.is_initialized:
                return
            
            print("🚀 Initializing Robust Browser Pool Manager...")
            
            # Check if browsers are installed
            if not browser_installation_state["is_installed"]:
                print("⚠️ Browsers not installed, attempting installation...")
                await self._install_browsers()
            
            try:
                self.playwright_instance = await async_playwright().start()
                
                # Enhanced browser launch args for cloud stability
                browser_args = [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                    '--disable-gpu-sandbox',
                    '--disable-software-rasterizer',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',
                    '--disable-javascript',  # We don't need JS for scraping
                    '--disable-default-apps',
                    '--disable-background-networking',
                    '--disable-sync',
                    '--no-default-browser-check',
                    '--memory-pressure-off',
                    '--max_old_space_size=512',
                    '--aggressive-cache-discard',
                    '--disable-hang-monitor',
                    '--disable-prompt-on-repost',
                    '--disable-client-side-phishing-detection',
                    '--disable-component-extensions-with-background-pages'
                ]
                
                self.browser = await self.playwright_instance.chromium.launch(
                    headless=True,
                    args=browser_args
                )
                
                # Test browser connection
                test_context = await self.browser.new_context()
                await test_context.close()
                
                self.is_initialized = True
                self.retry_count = 0
                print("✅ Robust Browser Pool Manager initialized successfully!")
                
            except Exception as e:
                print(f"❌ Failed to initialize Robust Browser Pool Manager: {e}")
                self.last_error = str(e)
                await self._cleanup()
                raise
    
    async def _install_browsers(self):
        """Install browsers if not available"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                browser_installation_state["is_installed"] = True
                print("✅ Browser installation successful")
            else:
                raise Exception(f"Browser installation failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Browser installation error: {e}")
            raise
    
    async def get_context(self) -> BrowserContext:
        """Get a new browser context with robust error handling and retries"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Ensure browser is initialized and connected
                if not self.is_initialized or not self.browser or not self.browser.is_connected():
                    print(f"🔄 Browser not ready (attempt {attempt + 1}), initializing...")
                    await self.initialize()
                
                # Create context with timeout
                context = await asyncio.wait_for(
                    self.browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        viewport={'width': 1280, 'height': 720},  # Smaller viewport for less memory
                        ignore_https_errors=True,
                        java_script_enabled=False,  # Disable JS for faster loading
                        extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'}
                    ),
                    timeout=30.0  # 30 second timeout for context creation
                )
                
                print(f"✅ Browser context created successfully (attempt {attempt + 1})")
                return context
                
            except asyncio.TimeoutError:
                print(f"⏱️ Context creation timeout (attempt {attempt + 1})")
                await self._handle_browser_failure()
                if attempt == max_attempts - 1:
                    raise Exception("Browser context creation timed out after all retries")
                    
            except Exception as e:
                print(f"⚠️ Error creating context (attempt {attempt + 1}): {e}")
                await self._handle_browser_failure()
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to create browser context after {max_attempts} attempts: {e}")
            
            # Exponential backoff
            wait_time = (2 ** attempt) + 1
            print(f"⏳ Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    async def _handle_browser_failure(self):
        """Handle browser failures with cleanup and reinitialize"""
        print("🔧 Handling browser failure...")
        await self._cleanup()
        self.retry_count += 1
        
        # If too many failures, wait longer
        if self.retry_count > 2:
            print(f"⚠️ Multiple browser failures ({self.retry_count}), waiting extra time...")
            await asyncio.sleep(5)
    
    async def _cleanup(self):
        """Enhanced cleanup with error handling"""
        try:
            if self.browser:
                try:
                    await asyncio.wait_for(self.browser.close(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("⏱️ Browser close timeout, forcing cleanup")
                except:
                    pass  # Ignore cleanup errors
                    
            if self.playwright_instance:
                try:
                    await asyncio.wait_for(self.playwright_instance.stop(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("⏱️ Playwright stop timeout")
                except:
                    pass  # Ignore cleanup errors
        except Exception as e:
            print(f"⚠️ Error during cleanup (ignored): {e}")
        finally:
            self.browser = None
            self.playwright_instance = None
            self.is_initialized = False
    
    async def close(self):
        """Close the browser pool with enhanced cleanup"""
        print("🔄 Closing Robust Browser Pool Manager...")
        await self._cleanup()
        print("✅ Robust Browser Pool Manager closed")

# Global robust browser pool manager
browser_pool = RobustBrowserPoolManager()

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
    """Update job progress in memory with error handling"""
    try:
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
        
        # Print progress for debugging
        print(f"📊 Job {job_id}: {status} - {progress}")
        
    except Exception as e:
        print(f"⚠️ Error updating job progress: {e}")

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

async def capture_page_screenshot_robust(page, url: str, topic: str) -> Optional[bytes]:
    """Enhanced screenshot capture with better error handling and timeouts"""
    try:
        print(f"📸 Capturing screenshot for URL: {url}")
        
        # Navigate with timeout and error handling
        try:
            await asyncio.wait_for(
                page.goto(url, wait_until="domcontentloaded", timeout=20000),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            print(f"⏱️ Navigation timeout for {url}")
            return None
        
        # Wait for page to settle
        await page.wait_for_timeout(2000)
        
        # Set smaller viewport for memory efficiency
        await page.set_viewport_size({"width": 1280, "height": 720})
        await page.wait_for_timeout(500)
        
        # Find all MCQ content elements with timeout
        mcq_elements = []
        
        try:
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
                
        except Exception as e:
            print(f"⚠️ Error finding elements: {e}")
            return None
        
        if not mcq_elements:
            print(f"❌ No MCQ elements found on {url}")
            return None
        
        # Calculate bounding box with error handling
        bounding_boxes = []
        for element in mcq_elements:
            try:
                box = await asyncio.wait_for(element.bounding_box(), timeout=5.0)
                if box:
                    bounding_boxes.append(box)
            except (asyncio.TimeoutError, Exception) as e:
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
        
        # Add padding and ensure reasonable bounds
        padding = 10  # Reduced padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        
        # Get viewport dimensions
        viewport = await page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
        screenshot_width = min(max_x - min_x + padding * 2, viewport['width'] - min_x, 1280)
        screenshot_height = min(max_y - min_y + padding * 2, viewport['height'] - min_y, 720)
        
        print(f"📐 Screenshot dimensions: {screenshot_width}x{screenshot_height} at ({min_x}, {min_y})")
        
        # Capture screenshot with timeout
        try:
            screenshot = await asyncio.wait_for(
                page.screenshot(
                    clip={
                        "x": min_x,
                        "y": min_y,
                        "width": screenshot_width,
                        "height": screenshot_height
                    },
                    type="png"
                ),
                timeout=15.0
            )
            
            print(f"✅ Screenshot captured successfully for {url}")
            return screenshot
            
        except asyncio.TimeoutError:
            print(f"⏱️ Screenshot capture timeout for {url}")
            return None
        
    except Exception as e:
        print(f"❌ Error capturing screenshot for {url}: {str(e)}")
        return None

async def scrape_testbook_page_with_screenshot_robust(context: BrowserContext, url: str, topic: str) -> Optional[dict]:
    """Enhanced screenshot scraping with robust error handling"""
    page = None
    try:
        print(f"🔍 Processing URL with screenshot (robust): {url}")
        
        # Create page with timeout
        try:
            page = await asyncio.wait_for(context.new_page(), timeout=10.0)
        except asyncio.TimeoutError:
            print(f"⏱️ Page creation timeout for {url}")
            return None
        
        # Check if page has MCQ content first (faster than full scraping)
        try:
            await asyncio.wait_for(
                page.goto(url, wait_until="domcontentloaded", timeout=20000),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            print(f"⏱️ Page load timeout for {url}")
            return None
        
        await page.wait_for_timeout(1000)
        
        # Quick relevance check
        try:
            question_element = await page.query_selector('h1.questionBody.tag-h1')
            if not question_element:
                question_element = await page.query_selector('div.questionBody')
            
            if not question_element:
                print(f"❌ No MCQ content found on {url}")
                return None
            
            # Extract question text for relevance check
            question_text = await question_element.inner_text()
            if not is_mcq_relevant(question_text, topic):
                print(f"❌ MCQ not relevant for topic '{topic}' on {url}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error during relevance check for {url}: {e}")
            return None
        
        # If relevant, capture screenshot
        screenshot = await capture_page_screenshot_robust(page, url, topic)
        
        if not screenshot:
            print(f"❌ Failed to capture screenshot for {url}")
            return None
        
        return {
            "url": url,
            "screenshot": screenshot,
            "is_relevant": True
        }
        
    except Exception as e:
        print(f"❌ Error processing {url} with screenshot: {str(e)}")
        return None
    finally:
        if page:
            try:
                await asyncio.wait_for(page.close(), timeout=5.0)
            except:
                pass  # Ignore close errors

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
        'reasoning': ['logical', 'logic', 'puzzle', 'pattern', 'sequence', 'analogy', 'verbal', 'analytical', 'solve', 'problem'],
        'cell': ['cellular', 'membrane', 'nucleus', 'mitosis', 'meiosis', 'organelle', 'cytoplasm', 'ribosome', 'mitochondria', 'chromosome'],
        'mitosis': ['cell', 'division', 'chromosome', 'spindle', 'kinetochore', 'centromere', 'anaphase', 'metaphase', 'prophase', 'telophase']
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
    max_results = 50  # Reduced for better performance
    
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

async def scrape_mcq_content_with_page_robust(page, url: str, search_topic: str) -> Optional[MCQData]:
    """Enhanced MCQ scraping with robust error handling and timeouts"""
    try:
        # Navigate with timeout
        try:
            await asyncio.wait_for(
                page.goto(url, wait_until='domcontentloaded', timeout=15000),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            print(f"⏱️ Navigation timeout for {url}")
            return None
        
        # Reduced wait time
        await page.wait_for_timeout(1000)
        
        # Extract question with timeout
        question = ""
        try:
            question_selectors = ['h1.questionBody.tag-h1', 'div.questionBody']
            for selector in question_selectors:
                try:
                    element = await asyncio.wait_for(page.query_selector(selector), timeout=5.0)
                    if element:
                        question = await asyncio.wait_for(element.inner_text(), timeout=5.0)
                        break
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"⚠️ Error extracting question from {url}: {e}")
            return None
        
        if not question:
            print(f"❌ No question found on {url}")
            return None
        
        # Clean question text
        question = clean_unwanted_text(question)
        
        # Check topic relevance
        print(f"🔍 DEBUG: Checking relevance for: '{question[:100]}...'")
        if not is_mcq_relevant(question, search_topic):
            print(f"❌ MCQ not relevant for topic '{search_topic}'")
            return None
        
        print(f"✅ MCQ relevant - topic '{search_topic}' found in question body")
        
        # Extract other elements with timeout (simplified)
        options = []
        answer = ""
        exam_source_heading = ""
        exam_source_title = ""
        
        try:
            # Options
            option_elements = await asyncio.wait_for(page.query_selector_all('li.option'), timeout=5.0)
            if option_elements:
                for option_elem in option_elements:
                    try:
                        option_text = await asyncio.wait_for(option_elem.inner_text(), timeout=3.0)
                        options.append(clean_unwanted_text(option_text.strip()))
                    except asyncio.TimeoutError:
                        continue
            
            # Answer
            answer_element = await asyncio.wait_for(page.query_selector('.solution'), timeout=3.0)
            if answer_element:
                answer = await asyncio.wait_for(answer_element.inner_text(), timeout=3.0)
                answer = clean_unwanted_text(answer)
            
            # Exam source (optional)
            try:
                exam_heading_element = await asyncio.wait_for(page.query_selector('div.pyp-heading'), timeout=2.0)
                if exam_heading_element:
                    exam_source_heading = await asyncio.wait_for(exam_heading_element.inner_text(), timeout=2.0)
                    exam_source_heading = clean_unwanted_text(exam_source_heading)
                
                exam_title_element = await asyncio.wait_for(page.query_selector('div.pyp-title.line-ellipsis'), timeout=2.0)
                if exam_title_element:
                    exam_source_title = await asyncio.wait_for(exam_title_element.inner_text(), timeout=2.0)
                    exam_source_title = clean_unwanted_text(exam_source_title)
            except asyncio.TimeoutError:
                pass  # Exam source is optional
                
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout extracting elements from {url}")
        
        # Return MCQ data if we have essential content
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

async def scrape_mcq_content_robust(url: str, search_topic: str) -> Optional[MCQData]:
    """Enhanced MCQ scraping with robust browser pool management"""
    context = None
    page = None
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            print(f"🔍 Scraping attempt {attempt + 1} for {url}")
            
            # Get context from robust browser pool
            try:
                context = await browser_pool.get_context()
            except Exception as e:
                print(f"⚠️ Failed to get browser context (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    return None
                await asyncio.sleep(2)
                continue
            
            # Create page with timeout
            try:
                page = await asyncio.wait_for(context.new_page(), timeout=10.0)
            except asyncio.TimeoutError:
                print(f"⏱️ Page creation timeout (attempt {attempt + 1})")
                if context:
                    await context.close()
                if attempt == max_attempts - 1:
                    return None
                await asyncio.sleep(2)
                continue
            
            # Scrape content
            result = await scrape_mcq_content_with_page_robust(page, url, search_topic)
            return result
            
        except Exception as e:
            print(f"❌ Error in scraping attempt {attempt + 1} for {url}: {e}")
            if attempt == max_attempts - 1:
                return None
            await asyncio.sleep(2)
        finally:
            # Clean up resources
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=5.0)
                except:
                    pass
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=5.0)
                except:
                    pass
    
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
            leading=18
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
        
        # Enhanced statistics section
        stats_data = [
            ['📊 Collection Statistics', ''],
            ['Search Topic', f'{topic}'],
            ['Total Relevant Questions', f'{len(mcqs)}'],
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
        story.append(Spacer(1, 0.4*inch))
        
        # Professional separator
        story.append(Paragraph("═" * 80, ParagraphStyle('separator', textColor=primary_color, alignment=TA_CENTER)))
        story.append(PageBreak())
        
        # Enhanced MCQ content
        for i, mcq in enumerate(mcqs, 1):
            # Professional question header
            story.append(Paragraph(f"QUESTION {i} OF {len(mcqs)}", question_header_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Exam source information
            if mcq.exam_source_heading or mcq.exam_source_title:
                exam_source_text = ""
                if mcq.exam_source_heading:
                    exam_source_text += f"📋 {mcq.exam_source_heading}"
                if mcq.exam_source_title:
                    exam_source_text += f" - {mcq.exam_source_title}"
                
                if exam_source_text:
                    story.append(Paragraph(exam_source_text, exam_source_style))
                    story.append(Spacer(1, 0.1*inch))
            
            # Question content
            question_text = mcq.question.replace('\n', '<br/>')
            story.append(Paragraph(f"<b>Q{i}:</b> {question_text}", question_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Options
            if mcq.options:
                story.append(Paragraph("📝 <b>OPTIONS:</b>", option_style))
                for j, option in enumerate(mcq.options):
                    option_letter = chr(ord('A') + j) if j < 26 else f"Option {j+1}"
                    option_text = option.replace('\n', '<br/>')
                    story.append(Paragraph(f"<b>{option_letter}.</b> {option_text}", option_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Answer
            if mcq.answer:
                story.append(Paragraph("💡 <b>ANSWER & DETAILED SOLUTION:</b>", answer_style))
                answer_text = mcq.answer.replace('\n', '<br/>')
                story.append(Paragraph(answer_text, answer_style))
            
            # Professional separator between questions
            story.append(Spacer(1, 0.25*inch))
            story.append(Paragraph("─" * 100, ParagraphStyle('divider', textColor=primary_color, alignment=TA_CENTER, fontSize=8)))
            story.append(Spacer(1, 0.25*inch))
            
            # Page break every 2 questions
            if i % 2 == 0 and i < len(mcqs):
                story.append(PageBreak())
        
        # Build PDF
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
            page_width = letter[0] - 2*inch
            page_height = letter[1] - 3*inch
            
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
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Image-based PDF generated successfully: {filename} with {len(screenshots_data)} screenshots")
        return filename
        
    except Exception as e:
        print(f"❌ Error generating image-based PDF: {e}")
        raise

async def process_mcq_extraction(job_id: str, topic: str, exam_type: str = "SSC", pdf_format: str = "text"):
    """Enhanced processing with robust error handling"""
    try:
        update_job_progress(job_id, "running", f"🔍 Searching for {exam_type} '{topic}' results with smart filtering...")
        
        # Search for links
        links = await search_google_custom(topic, exam_type)
        
        if not links:
            update_job_progress(job_id, "completed", f"❌ No {exam_type} results found for '{topic}'. Please try another topic.", 
                              total_links=0, processed_links=0, mcqs_found=0)
            return
        
        update_job_progress(job_id, "running", f"✅ Found {len(links)} {exam_type} links. Starting smart filtering extraction...", 
                          total_links=len(links))
        
        if pdf_format == "image":
            await process_screenshot_extraction_robust(job_id, topic, exam_type, links)
        else:
            await process_text_extraction_robust(job_id, topic, exam_type, links)
        
    except Exception as e:
        error_message = str(e)
        print(f"❌ Critical error in process_mcq_extraction: {e}")
        update_job_progress(job_id, "error", f"❌ Error: {error_message}")

async def process_text_extraction_robust(job_id: str, topic: str, exam_type: str, links: List[str]):
    """Robust text-based MCQ extraction"""
    try:
        # Initialize browser pool
        await browser_pool.initialize()
        
        mcqs = []
        relevant_mcqs = 0
        irrelevant_mcqs = 0
        
        print(f"🚀 Starting ROBUST text processing: {len(links)} links sequentially")
        
        # Process URLs one by one for maximum stability
        for i, url in enumerate(links):
            print(f"🔍 Processing link {i + 1}/{len(links)}: {url}")
            
            current_progress = f"🔍 Processing link {i + 1}/{len(links)} - Smart filtering enabled..."
            update_job_progress(job_id, "running", current_progress, 
                              processed_links=i, mcqs_found=len(mcqs))
            
            try:
                result = await scrape_mcq_content_robust(url, topic)
                
                if result:
                    mcqs.append(result)
                    relevant_mcqs += 1
                    print(f"✅ Found relevant MCQ {i + 1}/{len(links)} - Total: {len(mcqs)}")
                else:
                    irrelevant_mcqs += 1
                    print(f"⚠️ Skipped irrelevant MCQ {i + 1}/{len(links)}")
                    
            except Exception as e:
                print(f"❌ Error processing link {i + 1}: {e}")
                irrelevant_mcqs += 1
            
            # Update progress
            update_job_progress(job_id, "running", 
                              f"✅ Processed {i + 1}/{len(links)} links - Found {len(mcqs)} relevant MCQs", 
                              processed_links=i + 1, mcqs_found=len(mcqs))
            
            # Delay to prevent overload
            if i < len(links) - 1:
                await asyncio.sleep(1)
        
        # Clean up browser pool
        await browser_pool.close()
        
        if not mcqs:
            update_job_progress(job_id, "completed", 
                              f"❌ No relevant MCQs found for '{topic}' across {len(links)} links. Please try another topic.", 
                              total_links=len(links), processed_links=len(links), mcqs_found=0)
            return
        
        # Generate PDF
        final_message = f"✅ Smart filtering complete! Found {relevant_mcqs} relevant MCQs from {len(links)} total links."
        update_job_progress(job_id, "running", final_message + " Generating PDF...", 
                          total_links=len(links), processed_links=len(links), mcqs_found=len(mcqs))
        
        try:
            filename = generate_pdf(mcqs, topic, job_id, relevant_mcqs, irrelevant_mcqs, len(links))
            pdf_url = f"/api/download-pdf/{filename}"
            
            generated_pdfs[job_id] = {
                "filename": filename,
                "topic": topic,
                "exam_type": exam_type,
                "mcqs_count": len(mcqs),
                "generated_at": datetime.now()
            }
            
            success_message = f"🎉 SUCCESS! Generated PDF with {len(mcqs)} relevant MCQs for topic '{topic}'."
            update_job_progress(job_id, "completed", success_message, 
                              total_links=len(links), processed_links=len(links), 
                              mcqs_found=len(mcqs), pdf_url=pdf_url)
            
            print(f"✅ Job {job_id} completed successfully with {len(mcqs)} MCQs")
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            update_job_progress(job_id, "error", f"❌ Error generating PDF: {str(e)}")
    
    except Exception as e:
        print(f"❌ Critical error in text extraction: {e}")
        update_job_progress(job_id, "error", f"❌ Critical error: {str(e)}")
        await browser_pool.close()

async def process_screenshot_extraction_robust(job_id: str, topic: str, exam_type: str, links: List[str]):
    """Enhanced screenshot extraction with robust error handling"""
    try:
        # Initialize browser pool
        await browser_pool.initialize()
        
        screenshot_data = []
        relevant_mcqs = 0
        irrelevant_mcqs = 0
        
        print(f"🚀 Starting ROBUST screenshot processing: {len(links)} links")
        
        # Process URLs one by one for maximum stability
        for i, url in enumerate(links):
            print(f"📸 Processing screenshot {i + 1}/{len(links)}: {url}")
            
            current_progress = f"📸 Capturing screenshot {i + 1}/{len(links)} - Smart filtering enabled..."
            update_job_progress(job_id, "running", current_progress, 
                              processed_links=i, mcqs_found=len(screenshot_data))
            
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    # Get context with retries
                    context = await browser_pool.get_context()
                    result = await scrape_testbook_page_with_screenshot_robust(context, url, topic)
                    
                    if result and result.get('is_relevant'):
                        screenshot_data.append(result)
                        relevant_mcqs += 1
                        print(f"✅ Captured relevant screenshot {i + 1}/{len(links)} - Total: {len(screenshot_data)}")
                    else:
                        irrelevant_mcqs += 1
                        print(f"⚠️ Skipped irrelevant screenshot {i + 1}/{len(links)}")
                    
                    await context.close()
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"❌ Error capturing screenshot {i + 1} (attempt {attempt + 1}): {e}")
                    if attempt == max_attempts - 1:
                        irrelevant_mcqs += 1
                    else:
                        await asyncio.sleep(2)  # Wait before retry
            
            # Update progress
            update_job_progress(job_id, "running", 
                              f"✅ Processed {i + 1}/{len(links)} links - Captured {len(screenshot_data)} relevant screenshots", 
                              processed_links=i + 1, mcqs_found=len(screenshot_data))
            
            # Delay to prevent overload
            if i < len(links) - 1:
                await asyncio.sleep(2)
        
        # Clean up browser pool
        await browser_pool.close()
        
        if not screenshot_data:
            update_job_progress(job_id, "completed", 
                              f"❌ No relevant screenshots captured for '{topic}'. Please try another topic.", 
                              total_links=len(links), processed_links=len(links), mcqs_found=0)
            return
        
        # Generate image-based PDF
        try:
            final_message = f"✅ Screenshot capture complete! Captured {relevant_mcqs} relevant screenshots from {len(links)} total links."
            update_job_progress(job_id, "running", final_message + " Generating PDF...", 
                              total_links=len(links), processed_links=len(links), mcqs_found=len(screenshot_data))
            
            filename = generate_image_based_pdf(screenshot_data, topic, exam_type)
            pdf_url = f"/api/download-pdf/{filename}"
            
            generated_pdfs[job_id] = {
                "filename": filename,
                "topic": topic,
                "exam_type": exam_type,
                "mcqs_count": len(screenshot_data),
                "generated_at": datetime.now()
            }
            
            success_message = f"🎉 SUCCESS! Generated image-based PDF with {len(screenshot_data)} relevant screenshots for topic '{topic}'."
            update_job_progress(job_id, "completed", success_message, 
                              total_links=len(links), processed_links=len(links), 
                              mcqs_found=len(screenshot_data), pdf_url=pdf_url)
            
            print(f"✅ Screenshot job {job_id} completed successfully with {len(screenshot_data)} images")
            
        except Exception as e:
            print(f"❌ Error generating image PDF: {e}")
            update_job_progress(job_id, "error", f"❌ Error generating image PDF: {str(e)}")
    
    except Exception as e:
        print(f"❌ Critical error in screenshot extraction: {e}")
        update_job_progress(job_id, "error", f"❌ Critical error: {str(e)}")
        await browser_pool.close()

# API Routes
@app.get("/api/health")
async def health_check():
    """Enhanced health check with browser status"""
    return {
        "status": "healthy",
        "message": "MCQ Scraper API is running",
        "browser_status": {
            "installed": browser_installation_state["is_installed"],
            "installation_attempted": browser_installation_state["installation_attempted"],
            "installation_error": browser_installation_state.get("installation_error"),
            "browser_pool_initialized": browser_pool.is_initialized
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/generate-mcq-pdf")
async def generate_mcq_pdf(request: SearchRequest, background_tasks: BackgroundTasks):
    """Generate MCQ PDF with robust error handling"""
    job_id = str(uuid.uuid4())
    
    # Validate inputs
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")
    
    if request.exam_type not in ["SSC", "BPSC"]:
        raise HTTPException(status_code=400, detail="Exam type must be SSC or BPSC")
    
    if request.pdf_format not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="PDF format must be 'text' or 'image'")
    
    # Initialize job progress
    update_job_progress(
        job_id, 
        "running", 
        f"🚀 Starting {request.exam_type} MCQ extraction for '{request.topic}' ({request.pdf_format} format)..."
    )
    
    # Start background task
    background_tasks.add_task(
        process_mcq_extraction,
        job_id=job_id,
        topic=request.topic.strip(),
        exam_type=request.exam_type,
        pdf_format=request.pdf_format
    )
    
    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Started extracting {request.exam_type} MCQs for '{request.topic}' ({request.pdf_format} format)",
        "progress": f"🚀 Starting {request.exam_type} MCQ extraction for '{request.topic}' ({request.pdf_format} format)..."
    }

@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status with enhanced error handling"""
    try:
        if job_id not in job_progress:
            raise HTTPException(status_code=404, detail="Job not found")
        
        status = job_progress[job_id]
        print(f"📊 Returning status for job {job_id}: {status.get('status')} - {status.get('progress', '')[:100]}...")
        return status
        
    except Exception as e:
        print(f"❌ Error getting job status for {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "progress": f"Error retrieving job status: {str(e)}",
            "total_links": 0,
            "processed_links": 0,
            "mcqs_found": 0,
            "pdf_url": None
        }

@app.get("/api/download-pdf/{filename}")
async def download_pdf(filename: str):
    """Download generated PDF file"""
    filepath = f"/app/pdfs/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/pdf'
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Robust MCQ Scraper API starting up...")
    print(f"📊 Browser installation status: {browser_installation_state}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced cleanup on shutdown"""
    print("🔄 Robust MCQ Scraper API shutting down...")
    try:
        await browser_pool.close()
        print("✅ Browser pool closed successfully")
    except Exception as e:
        print(f"⚠️ Error closing browser pool: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
