from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.chunker import chunk_document

app = FastAPI(title="RAG Chunker Demo")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="sample"), name="samples")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chunk")
async def chunk_file(
    file: UploadFile = File(...),
    base_max_size: int = Form(1024),
    key_content_max_size: int = Form(1536),
    overlap: int = Form(100),
):
    if not file.filename:
        return JSONResponse({"error": "Missing filename."}, status_code=400)

    if base_max_size <= 0 or key_content_max_size <= 0 or overlap < 0:
        return JSONResponse({"error": "Invalid chunk parameters."}, status_code=400)

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            return JSONResponse(
                {"error": "Unsupported file encoding. Please upload UTF-8 or GBK text files."},
                status_code=400,
            )

    if not text.strip():
        return JSONResponse({"error": "The uploaded file is empty."}, status_code=400)

    data = chunk_document(
        text=text,
        base_max_size=base_max_size,
        key_content_max_size=key_content_max_size,
        overlap=overlap,
    )

    return {
        "filename": file.filename,
        "params": {
            "base_max_size": base_max_size,
            "key_content_max_size": key_content_max_size,
            "overlap": overlap,
        },
        "result": data,
    }
