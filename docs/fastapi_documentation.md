# FastAPI Technical Documentation

## Overview

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.11+ based on standard Python type hints. It is one of the fastest Python frameworks available, on par with NodeJS and Go, thanks to Starlette and Pydantic.

**Key Features:**
- Fast: Very high performance, on par with NodeJS and Go
- Fast to code: Increase development speed by 200-300%
- Fewer bugs: Reduce human-induced errors by 40%
- Intuitive: Great editor support with auto-completion everywhere
- Easy: Designed to be easy to use and learn
- Short: Minimize code duplication
- Robust: Get production-ready code with automatic interactive documentation
- Standards-based: Based on OpenAPI and JSON Schema

---

## Installation

Install FastAPI and Uvicorn (ASGI server):

```bash
pip install fastapi uvicorn[standard]
```

Or using uv:
```bash
uv add fastapi uvicorn
```

---

## Creating Your First FastAPI App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Run the server:
```bash
uvicorn main:app --reload
```

---

## Path Parameters

Path parameters are declared using Python type annotations:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

@app.get("/items/{item_id}")
def get_item(item_id: str):
    return {"item_id": item_id}
```

FastAPI automatically validates and converts the path parameter to the declared type. If you declare `user_id: int` and a non-integer is passed, a 422 Unprocessable Entity error is returned automatically.

---

## Query Parameters

Parameters not declared as path parameters are treated as query parameters:

```python
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

Query parameters can be optional with a default value:

```python
@app.get("/items/{item_id}")
def read_item(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "This item has a long description"})
    return item
```

---

## Request Body with Pydantic

Use Pydantic models to declare request bodies:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

---

## Response Models

Declare the response model with `response_model`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ItemInput(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class ItemOutput(BaseModel):
    name: str
    price: float

@app.post("/items/", response_model=ItemOutput)
def create_item(item: ItemInput):
    return item  # FastAPI will filter to only ItemOutput fields
```

Using `response_model` ensures:
1. Output data is validated against the model
2. Sensitive fields not in the response model are automatically excluded
3. OpenAPI documentation is generated for the response

---

## HTTP Status Codes

Specify the HTTP status code for a response:

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(name: str):
    return {"name": name}

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    return None
```

Common status codes:
- 200 OK: Default for GET requests
- 201 Created: Resource successfully created
- 204 No Content: Success with no body (DELETE)
- 400 Bad Request: Client error
- 401 Unauthorized: Authentication required
- 403 Forbidden: Insufficient permissions
- 404 Not Found: Resource doesn't exist
- 422 Unprocessable Entity: Validation error (automatic)
- 500 Internal Server Error: Server-side error

---

## Error Handling with HTTPException

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail=f"Item '{item_id}' not found"
        )
    return {"item": items[item_id]}
```

### Custom Exception Handlers

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )
```

---

## Dependency Injection

FastAPI has a powerful dependency injection system:

```python
from fastapi import FastAPI, Depends

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

Dependencies can be:
- Functions (sync or async)
- Classes with `__call__`
- Other FastAPI dependencies (nested)
- Path operations (used as decorators)

---

## CORS Middleware

Add Cross-Origin Resource Sharing (CORS) support:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://myapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],        # or ["GET", "POST"]
    allow_headers=["*"],
)

@app.get("/")
def main():
    return {"message": "Hello World"}
```

**Security note**: Using `allow_origins=["*"]` with `allow_credentials=True` is not allowed by browsers. If you need credentials, specify explicit origins.

---

## Authentication with OAuth2

FastAPI provides utilities for OAuth2 with Password Bearer:

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return decode_token(token)
```

---

## Background Tasks

Run tasks after returning a response:

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="a") as email_file:
        content = f"notification for {email}: {message}\n"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in background"}
```

Background tasks are useful for:
- Sending emails after user registration
- Processing files after upload
- Sending webhooks after state changes
- Logging analytics events

---

## File Upload

Handle file uploads with FastAPI:

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
    }

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}
```

For large files, use `UploadFile` (streaming) instead of `bytes` (loads entirely into memory).

---

## Async Endpoints

FastAPI supports both sync and async endpoint functions:

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

# Sync endpoint (runs in thread pool executor)
@app.get("/sync")
def sync_endpoint():
    return {"type": "synchronous"}

# Async endpoint (runs in event loop directly)
@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(0)  # Example: await DB call
    return {"type": "asynchronous"}
```

**Rule of thumb**: 
- Use `async def` for IO-bound operations (database queries, HTTP calls, file reads)
- Use `def` for CPU-bound or synchronous library calls (they run in a thread pool)

---

## Middleware

Middleware runs before and after every request:

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## Lifespan Events

Run code at startup and shutdown:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs at STARTUP
    print("Loading ML model...")
    load_model()
    yield
    # Code here runs at SHUTDOWN
    print("Cleaning up resources...")
    cleanup()

app = FastAPI(lifespan=lifespan)
```

Use lifespan for:
- Loading ML models into memory
- Connecting to databases
- Initializing caches
- Warming up external services

---

## OpenAPI Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI JSON**: Available at `/openapi.json`

Customize the docs:
```python
app = FastAPI(
    title="My API",
    description="This is a very fancy API",
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    docs_url="/documentation",
    redoc_url="/redocumentation",
)
```

---

## Environment Variables with Pydantic Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My App"
    database_url: str
    secret_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI()

@app.get("/info")
async def info():
    return {"app_name": settings.app_name}
```

---

## Testing FastAPI Apps

Use `TestClient` from Starlette for testing:

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Foo", "price": 10.5},
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Foo"
```

---

## Common Error: 422 Unprocessable Entity

The 422 error occurs when request validation fails. Common causes:

1. **Wrong data type**: Sending a string where an int is expected
2. **Missing required field**: Not providing a required request body field
3. **Invalid enum value**: Providing a value not in a defined enum
4. **Failed regex validation**: String doesn't match the defined pattern

Example response:
```json
{
    "detail": [
        {
            "loc": ["body", "price"],
            "msg": "value is not a valid float",
            "type": "type_error.float"
        }
    ]
}
```

To fix: Check the `detail` array for the field name and error type.

---

## Performance Tips

1. **Use async functions** for I/O operations (DB queries, HTTP calls)
2. **Connection pooling**: Use SQLAlchemy's connection pool for databases
3. **Caching**: Use `functools.lru_cache` or Redis for expensive computations
4. **Response compression**: Add GZip middleware for large responses
5. **Database indexing**: Ensure your queries use indexed columns
6. **Pagination**: Always paginate large list responses

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```
