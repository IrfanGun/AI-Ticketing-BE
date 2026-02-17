import logging
import os
import uuid
import json
import time
import re
from typing import Optional, Literal, Dict, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request
from pydantic import BaseModel, Field, conint
from sqlalchemy import create_engine, Column, String, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# 1. Load environment variables dari file .env
load_dotenv()

# 2. Database Configuration
# GANTI "isi_link_neon_kamu" dengan link postgresql dari dashboard Neon.tech
DATABASE_URL = os.getenv("DATABASE_URL") or "isi_link_neon_kamu"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. Model Database (Tabel Tiket)
class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(String, primary_key=True, index=True)
    message = Column(Text)
    status = Column(String, default="processing")
    category = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    ai_draft = Column(Text, nullable=True)
    sentiment_score = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

# Buat tabel otomatis jika belum ada di Neon
Base.metadata.create_all(bind=engine)

# 4. FastAPI Setup
app = FastAPI(title="AI Support Triage System")

# Agar frontend (Next.js) bisa mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. AI Client Setup (Groq)
AI_API_KEY = os.getenv("GROQ_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-oss-safeguard-20b")
if not AI_API_KEY:
    raise RuntimeError("AI API key missing: set GROQ_API_KEY.")

ai_client = Groq(api_key=AI_API_KEY)

# 6. Schema Data (Diletakkan di atas agar tidak NameError)
class TicketRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

class TicketUpdate(BaseModel):
    ai_draft: Optional[str] = None
    status: Optional[str] = Field(default=None, pattern="^(processing|completed|failed|resolved)$")

class TicketResponse(BaseModel):
    id: str
    message: str
    status: str
    category: Optional[str]
    urgency: Optional[str]
    sentiment_score: Optional[int]
    ai_draft: Optional[str]
    error_message: Optional[str]

class AIResult(BaseModel):
    category: Literal["Billing", "Technical", "Feature Request"]
    urgency: Literal["High", "Medium", "Low"]
    sentiment: conint(ge=1, le=10)
    draft_response: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-triage")

# Keamanan & rate limit sederhana
API_KEY = os.getenv("API_KEY")
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_requests_log: Dict[str, List[float]] = {}


def rate_limiter(client_id: str):
    now = time.time()
    window_start = now - 60
    history = _requests_log.get(client_id, [])
    history = [t for t in history if t >= window_start]
    if len(history) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    history.append(now)
    _requests_log[client_id] = history


def require_api_key(request: Request):
    if API_KEY is None:
        # Jika API_KEY tidak diset, biarkan terbuka (development)
        return
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    # gunakan IP sebagai identitas untuk rate limit
    client_id = request.client.host if request.client else "unknown"
    rate_limiter(client_id)


def extract_json_block(text: str) -> str:
    """Ambil konten JSON dari balasan yang mungkin dibungkus ```json ...```."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return text

# 7. LOGIKA AI (Proses di belakang layar agar user tidak menunggu)
def process_ai_analysis(ticket_id: str, message: str):
    db = SessionLocal()
    try:
        prompt = (
            "You are an AI triage assistant. Read the customer message below and return ONLY a JSON object.\n"
            f"Message: '''{message}'''\n"
            "Fields:\n"
            "category: one of [Billing, Technical, Feature Request]\n"
            "urgency: one of [High, Medium, Low]\n"
            "sentiment: integer 1-10 (1 very negative, 10 very positive)\n"
            "draft_response: concise, polite, context-aware reply in Indonesian.\n"
            "Do not include extra keys or text."
        )
        last_error = None
        for attempt in range(3):
            try:
                completion = ai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model=AI_MODEL,
                    timeout=20,
                )
                ai_content = extract_json_block(completion.choices[0].message.content)
                try:
                    ai_results = AIResult.model_validate_json(ai_content)
                except Exception:
                    ai_results = AIResult.model_validate_json(json.dumps(json.loads(ai_content)))

                ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
                if ticket:
                    ticket.category = ai_results.category
                    ticket.urgency = ai_results.urgency
                    ticket.sentiment_score = ai_results.sentiment
                    ticket.ai_draft = ai_results.draft_response
                    ticket.status = "completed"
                    ticket.error_message = None
                    db.commit()
                return
            except Exception as inner_err:
                last_error = inner_err
                time.sleep(1 + attempt)
                continue
        # jika semua percobaan gagal
        raise last_error or RuntimeError("Unknown AI error")

    except Exception as e:
        logger.exception("Error processing AI for ticket %s", ticket_id)
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket:
            ticket.status = "failed"
            ticket.error_message = str(e)
            db.commit()
    finally:
        db.close()
# 8. ENDPOINTS (Jalur masuk data)

@app.post("/tickets", status_code=201)
async def create_ticket(
    request: TicketRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
):
    ticket_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        # Simpan pesan asli ke database
        new_ticket = Ticket(id=ticket_id, message=request.message)
        db.add(new_ticket)
        db.commit()
        
        # Jalankan analisis AI sebagai background task
        background_tasks.add_task(process_ai_analysis, ticket_id, request.message)
        
        return {"id": ticket_id, "status": "accepted"}
    finally:
        db.close()

@app.get("/tickets")
async def get_all_tickets(
    limit: int = 100,
    offset: int = 0,
    _: None = Depends(require_api_key),
):
    db = SessionLocal()
    tickets = db.query(Ticket).offset(offset).limit(limit).all()
    db.close()
    return tickets

@app.put("/tickets/{ticket_id}")
async def update_ticket(ticket_id: str, body: TicketUpdate, _: None = Depends(require_api_key)):
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if body.ai_draft is not None:
            ticket.ai_draft = body.ai_draft
        if body.status is not None:
            ticket.status = body.status
        db.commit()
        db.refresh(ticket)
        return TicketResponse(
            id=ticket.id,
            message=ticket.message,
            status=ticket.status,
            category=ticket.category,
            urgency=ticket.urgency,
            sentiment_score=ticket.sentiment_score,
            ai_draft=ticket.ai_draft,
            error_message=ticket.error_message,
        )
    finally:
        db.close()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        return {"status": "ready"}
    finally:
        db.close()
