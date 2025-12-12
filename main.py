from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette import status
import uvicorn
import os
import sqlite3
import hashlib # Using standard hashlib for secure hashing (SHA-256)

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
    # FIX: check_same_thread=False is essential for SQLite in a multi-threaded web server
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

def create_db_table():
    """
    Creates the users and doctors tables.
    The explicit DROP TABLE commands have been removed to ensure data persistence.
    """
    print(f"Checking/Creating database file: {DATABASE_FILE}")
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    try:
        # --- DEVELOPMENT SCHEMA RESET START ---
        # Dropping existing tables to apply new schema (role column)
        # conn.execute("DROP TABLE IF EXISTS users") <-- REMOVED FOR PERSISTENCE
        # conn.execute("DROP TABLE IF EXISTS doctors") <-- REMOVED FOR PERSISTENCE
        # --- DEVELOPMENT SCHEMA RESET END ---

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
        
        # 2. Doctors Table (Created but unused for primary insertion, kept for potential future use)
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
    # Query the USERS table for the max UID
    cursor = db.execute("SELECT MAX(uid) FROM users").fetchone()
    
    # FIX: Ensure max_uid is treated as 0 if the query returns NULL (i.e., cursor[0] is None)
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
if not os.path.exists("templates"):
    os.makedirs("templates")

# Configure templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Utility Context for Templates ---

def get_template_context(request: Request, user_name: str = "Anonymous"):
    """Returns the base context required by Jinja2 templates."""
    error = request.query_params.get("error")
    return {"request": request, "user_name": user_name, "error": error}

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
    role: str = Form(...) # Capture the role submitted by the switch
):
    """Handles user login, checking against the USERS table and verifying role."""
    
    # Query the single USERS table
    cursor = db.execute(
        "SELECT uid, password, name, role FROM users WHERE email = ?",
        (email,)
    )
    user = cursor.fetchone()
    
    # 1. Check if user exists and password is correct
    if user and verify_password(password, user['password']):
        # 2. Check if the stored role matches the submitted role
        if user['role'] == role:
            print(f"User logged in: UID {user['uid']}, Role: {user['role']}")
            
            # SUCCESS: Redirect based on the role stored in the database
            redirect_path = "/doctor_dashboard" if user['role'] == 'doctor' else "/dashboard"
            return RedirectResponse(f"{redirect_path}?uid={user['uid']}", status_code=status.HTTP_303_SEE_OTHER)
        else:
            error_message = f"Role mismatch. Please confirm you are logging in as a {user['role']}."
            print(f"Login failed: Role mismatch for {email}. Stored role: {user['role']}, Submitted role: {role}")
    else:
        # Invalid credentials error
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

    # 1. Check if passwords match
    if password != confirm_password:
        return JSONResponse(
            {"message": "Passwords do not match."},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Role validation and table assignment (always 'users')
    if role not in ['user', 'doctor']:
        role = 'user'
    
    redirect_path = '/doctor_dashboard' if role == 'doctor' else '/dashboard'
    table_name = 'users' # Always insert into 'users' table

    try:
        # 2. Hash the password securely
        password_hash = get_password_hash(password)
        
        # 3. Get the next unique UID 
        next_uid = get_next_uid(db) 
        
        # INSERT into the USERS table
        db.execute(
            f"INSERT INTO {table_name} (uid, name, email, phone, password, role) VALUES (?, ?, ?, ?, ?, ?)",
            (next_uid, name, email, phone, password_hash, role)
        )
        db.commit()
        
        print(f"New user registered: UID {next_uid}, Email: {email}, Role: {role}")
        
        # 4. SUCCESS: Return JSON with correct UID and role-specific redirect path
        return JSONResponse(
            {"message": "Registration successful. Redirecting...", "redirect_url": f"{redirect_path}?uid={next_uid}"},
            status_code=status.HTTP_201_CREATED
        )

    except sqlite3.IntegrityError:
        # This will fail if the email already exists in the USERS table.
        return JSONResponse(
            {"message": "This email is already registered in the system. Please login instead."},
            status_code=status.HTTP_409_CONFLICT
        )

    except Exception as e:
        # Generic error handling
        print(f"!!! CRITICAL SERVER CRASH: {e}")
        return JSONResponse(
            {"message": "An unexpected server error occurred."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# --- APPLICATION ROUTES (Restored) ---

@app.get("/dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_dashboard(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
    uid: int | None = None
):
    """User/Patient Dashboard. Fetches user name from the USERS table."""
    
    user_name = "Anonymous"
    if uid:
        # Query the single USERS table
        cursor = db.execute("SELECT name FROM users WHERE uid = ?", (uid,))
        user = cursor.fetchone()
        if user:
            user_name = user['name']

    context = get_template_context(request, user_name=user_name)
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/doctor_dashboard", response_class=HTMLResponse, tags=["Views"])
async def read_doctor_dashboard(
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
    uid: int | None = None
):
    """Doctor/Provider Dashboard. Fetches name from the USERS table."""
    
    user_name = "Anonymous"
    if uid:
        # Query the single USERS table
        cursor = db.execute("SELECT name FROM users WHERE uid = ?", (uid,))
        user = cursor.fetchone()
        if user:
            user_name = user['name']

    context = get_template_context(request, user_name=user_name)
    return templates.TemplateResponse("doctor_dashboard.html", context)


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
