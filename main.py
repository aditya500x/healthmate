from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# --- FastAPI Initialization ---
app = FastAPI(title="HealthMate AI")

# Create a dummy 'static' directory if it doesn't exist (for serving static assets like images/CSS in a real deployment)
if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static directory to serve files like CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates directory
# NOTE: The HTML files must be inside a folder named 'templates'
templates = Jinja2Templates(directory="templates")

# --- Authentication/User Mock Data (Used for demonstration) ---
MOCK_USER = "Dr. Healthmate"

# --- Utility Context for Templates (Inject common data into all templates) ---
def get_template_context(request: Request):
    """Returns the base context required by Jinja2 templates."""
    return {"request": request, "user_name": MOCK_USER}

# --- Routes ---

@app.get("/", response_class=HTMLResponse, tags=["Views"])
async def read_root(request: Request):
    """Landing page view (Login/Marketing)."""
    context = get_template_context(request)
    return templates.TemplateResponse("index.html", context)

@app.get("/dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_dashboard(request: Request):
    """Main application dashboard view (Service Selection)."""
    context = get_template_context(request)
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/signup", response_class=HTMLResponse, tags=["Views"])
async def read_signup(request: Request):
    """User registration page."""
    context = get_template_context(request)
    return templates.TemplateResponse("signup.html", context)

# Route for the Login Page
@app.get("/login", response_class=HTMLResponse, tags=["Views"])
async def read_login(request: Request):
    """User login page."""
    context = get_template_context(request)
    return templates.TemplateResponse("login.html", context)


# If you run this file directly, it will start the Uvicorn server
if __name__ == "__main__":
    # Ensure the templates directory exists for local testing
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created 'templates' directory. Please save HTML files there.")
        
    print("Starting HealthMate AI server on http://127.0.0.1:8000")
    print("Visit http://127.0.0.1:8000/, http://127.0.0.1:8000/dashboard, http://127.0.0.1:8000/signup, and http://127.0.0.1:8000/login")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
