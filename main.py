from fastapi import FastAPI, Request, Form, Depends, UploadFile, File 
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette import status
import uvicorn
import os
import sqlite3
import hashlib 
import shutil 
import easyocr 
from analizerend.analizer import analyze_prescription_image 

# --- GLOBAL MODEL INITIALIZATION CHECK ---
print("Checking status of analyzerend module...")
try:
    from analizerend.analizer import READER 
    if READER is None:
        print("WARNING: OCR Engine failed to load in analizerend. Analysis will likely fail.")
    else:
        print("ANALYZEREND module is successfully imported and initialized.")
except ImportError:
    print("CRITICAL ERROR: 'analizerend.analizer' module not found. AI features disabled.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to check analyzer status: {e}")
# -----------------------------------


# --- Database Configuration ---
DATABASE_FILE = "healthmate.db"
STARTING_UID = 10000

# --- Security Configuration (Using SHA-256) ---

def get_password_hash(password: str) -> str:
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored hash."""
    return get_password_hash(plain_password) == hashed_password

def get_db():
    """Dependency to get a database connection."""
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  
    try:
        yield conn
    finally:
        conn.close()

def create_db_table():
    """
    Creates the users and doctors tables (users table holds all data).
    Data persistence is maintained (no DROP TABLE commands).
    """
    print(f"Checking/Creating database file: {DATABASE_FILE}")
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    try:
        # 1. Users Table (Patients & Doctors - Single Source of Truth)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                uid INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)
        
        # 2. Doctors Table (Kept for potential future use, but not for primary data insertion)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY,
                uid INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'doctor'
            )
        """)
        
        conn.commit()
    finally:
        conn.close()

def get_next_uid(db: sqlite3.Connection) -> int:
    """Calculates the next sequential user ID (uid) based on the USERS table."""
    cursor = db.execute("SELECT MAX(uid) FROM users").fetchone()
    max_uid = cursor[0] if cursor and cursor[0] is not None else 0
    if max_uid < STARTING_UID:
        return STARTING_UID
    return max_uid + 1

# --- FastAPI Initialization ---
app = FastAPI(title="HealthMate AI")

# Initialize database
create_db_table()

# Ensure directories exist
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("uploads"): # Directory for temporary file uploads
    os.makedirs("uploads")
if not os.path.exists("templates"):
    os.makedirs("templates")

# Configure templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Utility Context for Templates ---
# ADDED uid to context
def get_template_context(request: Request, user_name: str = "Anonymous", uid: int | None = None):
    """Returns the base context required by Jinja2 templates."""
    error = request.query_params.get("error")
    # NEW: Pass UID in context
    return {"request": request, "user_name": user_name, "uid": uid, "error": error}

# --- API ENDPOINT FOR AI/ML FEATURE ---

@app.post("/api/analyze-prescription")
async def analyze_prescription_endpoint(file: UploadFile = File(...)):
    """Handles the file upload and runs the prescription analysis via the external module."""
    
    file_location = f"uploads/{file.filename}"
    
    try:
        # 1. Save the uploaded file to disk
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Run the external analysis function
        medications = analyze_prescription_image(file_location)
        
        # 3. Return successful JSON response
        return JSONResponse(
            {"message": "Analysis complete.", "medications": medications},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        print(f"Error processing prescription file: {e}")
        return JSONResponse(
            {"message": f"Analysis failed: {e}"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        # 4. Clean up temporary file
        if os.path.exists(file_location):
            os.remove(file_location)


# --- Core Routes ---

@app.get("/", response_class=HTMLResponse, tags=["Views"])
async def read_root(request: Request):
    """Landing page view (index.html)."""
    context = get_template_context(request)
    return templates.TemplateResponse("index.html", context)

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
    role: str = Form(...) 
):
    """Handles user login, checking against the USERS table and verifying role."""
    
    cursor = db.execute(
        "SELECT uid, password, name, role FROM users WHERE email = ?",
        (email,)
    )
    user = cursor.fetchone()
    
    if user and verify_password(password, user['password']):
        if user['role'] == role:
            print(f"User logged in: UID {user['uid']}, Role: {user['role']}")
            redirect_path = "/doctor_dashboard" if user['role'] == 'doctor' else "/dashboard"
            return RedirectResponse(f"{redirect_path}?uid={user['uid']}", status_code=status.HTTP_303_SEE_OTHER)
        else:
            error_message = f"Role mismatch. Please confirm you are logging in as a {user['role']}."
            print(f"Login failed: Role mismatch for {email}. Stored role: {user['role']}, Submitted role: {role}")
    else:
        error_message = "Invalid email or password."

    return RedirectResponse(f"/login?error={error_message}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/signup", response_class=HTMLResponse, tags=["Views"])
async def read_signup(request: Request):
    """User registration page."""
    context = get_template_context(request)
    return templates.TemplateResponse("signup.html", context)

@app.post("/signup")
async def signup_user(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Handles user signup: inserts ALL data into the 'users' table.
    """
    
    try:
        data = await request.json()
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        role = data.get('role', 'user') 

    except Exception:
        return JSONResponse(
            {"message": "Invalid data format."},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    if password != confirm_password:
        return JSONResponse(
            {"message": "Passwords do not match."},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    if role not in ['user', 'doctor']:
        role = 'user'
    
    redirect_path = '/doctor_dashboard' if role == 'doctor' else '/dashboard'
    table_name = 'users' 

    try:
        password_hash = get_password_hash(password)
        next_uid = get_next_uid(db) 
        
        db.execute(
            f"INSERT INTO {table_name} (uid, name, email, phone, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            (next_uid, name, email, phone, password_hash, role)
        )
        db.commit()
        
        print(f"New user registered: UID {next_uid}, Email: {email}, Role: {role}")
        
        return JSONResponse(
            {"message": "Registration successful. Redirecting...", "redirect_url": f"{redirect_path}?uid={next_uid}"},
            status_code=status.HTTP_201_CREATED
        )

    except sqlite3.IntegrityError:
        return JSONResponse(
            {"message": "This email is already registered in the system. Please login instead."},
            status_code=status.HTTP_409_CONFLICT
        )

    except Exception as e:
        print(f"!!! CRITICAL SERVER CRASH: {e}")
        return JSONResponse(
            {"message": "An unexpected server error occurred."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_dashboard(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
    uid: int | None = None
):
    """User/Patient Dashboard. Fetches user name and passes UID."""
    
    user_name = "Anonymous"
    user_uid = uid
    if uid:
        cursor = db.execute("SELECT name FROM users WHERE uid = ?", (uid,))
        user = cursor.fetchone()
        if user:
            user_name = user['name']

    # NEW: Pass the uid into the context
    context = get_template_context(request, user_name=user_name, uid=user_uid)
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/doctor_dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_doctor_dashboard(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
    uid: int | None = None
):
    """Doctor/Provider Dashboard. Fetches name and passes UID."""
    
    user_name = "Anonymous"
    user_uid = uid
    if uid:
        cursor = db.execute("SELECT name FROM users WHERE uid = ?", (uid,))
        user = cursor.fetchone()
        if user:
            user_name = user['name']

    # NEW: Pass the uid into the context
    context = get_template_context(request, user_name=user_name, uid=user_uid)
    return templates.TemplateResponse("doctor_dashboard.html", context)


@app.get("/prescription", response_class=HTMLResponse, tags=["Views"])
async def read_prescription_analysis(request: Request):
    """Prescription Analysis tool page. Expects UID via URL for context."""
    # Retrieve UID and Name if present in the URL query string (passed from dashboard)
    user_name = request.query_params.get("user_name", "Anonymous")
    uid = request.query_params.get("uid")
    
    context = get_template_context(request, user_name=user_name, uid=uid)
    return templates.TemplateResponse("prescription.html", context)

@app.get("/diet", response_class=HTMLResponse, tags=["Views"])
async def read_diet_plan(request: Request):
    """Diet Plan and tracking page."""
    context = get_template_context(request)
    return templates.TemplateResponse("diet.html", context)

@app.get("/lifestyle", response_class=HTMLResponse, tags=["Views"])
async def read_lifestyle_tracker(request: Request):
    """Lifestyle tracking and goal monitoring page."""
    context = get_template_context(request)
    return templates.TemplateResponse("lifestyle.html", context)

@app.get("/contact", response_class=HTMLResponse, tags=["Views"])
async def read_contact_page(request: Request):
    """Secure Messaging and Contact page."""
    context = get_template_context(request)
    return templates.TemplateResponse("contacts.html", context)


@app.get("/learn", response_class=HTMLResponse, tags=["Views"])
async def read_learn_more(request: Request):
    """Learn More informational page."""
    context = get_template_context(request)
    return templates.TemplateResponse("learn.html", context)


if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
