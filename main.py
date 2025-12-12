from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette import status
import uvicorn
import os
import sqlite3
import hashlib

# --- Database Configuration ---
DATABASE_FILE = "healthmate.db"
STARTING_UID = 10000

def get_db():
    """Dependency to get a database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

def create_db_table():
    """Creates the users table if it doesn't exist, using a direct connection."""
    print(f"Checking/Creating database file: {DATABASE_FILE}")
    conn = sqlite3.connect(DATABASE_FILE)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                uid INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()

def get_next_uid(db: sqlite3.Connection) -> int:
    """Calculates the next sequential user ID (uid) starting from STARTING_UID."""
    cursor = db.execute("SELECT MAX(uid) FROM users")
    max_uid = cursor.fetchone()[0]
    
    if max_uid is None:
        return STARTING_UID
    return max_uid + 1

# --- FastAPI Initialization ---
app = FastAPI(title="HealthMate AI")

# FIX: Call create_db_table directly outside the generator dependency context
create_db_table()

# Create a dummy 'static' directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Configure templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static directory to serve files like CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    # NOTE: This assumes an index.html file exists in the templates directory
    return templates.TemplateResponse("index.html", context)

# --- USER AUTH ROUTES ---

@app.get("/login", response_class=HTMLResponse, tags=["Views"])
async def read_login(request: Request):
    """User login page."""
    context = get_template_context(request)
    return templates.TemplateResponse("login.html", context)

@app.post("/login")
async def login_user(
    db: sqlite3.Connection = Depends(get_db),
    email: str = Form(...),
    password: str = Form(...),
):
    """Handles user login form submission and checks credentials."""
    
    # Hash the submitted password using MD5
    password_hash = hashlib.md5(password.encode('utf-8')).hexdigest()
    
    # Query the database for the user
    cursor = db.execute(
        "SELECT uid FROM users WHERE email = ? AND password = ?",
        (email, password_hash)
    )
    user = cursor.fetchone()
    
    if user:
        print(f"User logged in: UID {user['uid']}")
        # SUCCESS: In a real app, you would set a session or cookie here.
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    else:
        # FAILURE
        print(f"Login failed for email: {email}")
        # Redirect back to login page (can add error message in a future iteration)
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/signup", response_class=HTMLResponse, tags=["Views"])
async def read_signup(request: Request, error: str = None):
    """User registration page."""
    context = get_template_context(request)
    context['error'] = error
    return templates.TemplateResponse("signup.html", context)

@app.post("/signup")
async def signup_user(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Handles user signup form submission and saves user data to SQLite."""
    
    try:
        # Hash the password using MD5
        password_hash = hashlib.md5(password.encode('utf-8')).hexdigest()
        
        # Get the next UID
        next_uid = get_next_uid(db)
        
        db.execute(
            "INSERT INTO users (uid, name, email, phone, password) VALUES (?, ?, ?, ?, ?)",
            (next_uid, name, email, phone, password_hash)
        )
        db.commit()
        
        print(f"New user registered: UID {next_uid}, Email: {email}")
        
        # FIXED REDIRECTION: Redirect to dashboard on successful signup
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        print(f"Unexpected error during signup: {e}")
        context = get_template_context(request)
        context['error'] = "An unexpected error occurred."
        return templates.TemplateResponse("signup.html", context, status_code=500)


# --- APPLICATION ROUTES (Require Authentication in a real app) ---

@app.get("/dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_dashboard(request: Request):
    """Main application dashboard view (Service Selection)."""
    context = get_template_context(request)
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/prescription", response_class=HTMLResponse, tags=["Views"])
async def read_prescription_analysis(request: Request):
    """Prescription Analysis tool page."""
    context = get_template_context(request)
    return templates.TemplateResponse("prescription.html", context)

@app.get("/diet", response_class=HTMLResponse, tags=["Views"])
async def read_diet_plan(request: Request):
    """Diet Plan and tracking page."""
    context = get_template_context(request)
    return templates.TemplateResponse("diet.html", context)

@app.get("/learn", response_class=HTMLResponse, tags=["Views"])
async def read_learn_more(request: Request):
    """Learn More informational page."""
    context = get_template_context(request)
    return templates.TemplateResponse("learn.html", context)


# If you run this file directly, it will start the Uvicorn server
if __name__ == "__main__":
    # Ensure the templates directory exists for local testing
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created 'templates' directory. Please save HTML files there.")
        
    print("Starting HealthMate AI server on http://127.0.0.1:8000")
    print("Visit http://127.0.0.1:8000/")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
