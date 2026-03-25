from fastapi import HTTPException
from groq import Groq

from app.core.settings import Settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse


class AIChatService:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        if request.type != "financial_chat":
            raise HTTPException(status_code=422, detail=f"Unsupported analysis type: {request.type}")

        prompt = str(request.data.get("prompt", "")).strip()
        context_summary = str(request.data.get("context_summary", "")).strip()
        finance_context = request.data.get("finance_context", {})

        if not prompt:
            raise HTTPException(status_code=422, detail="Prompt is required")

        user_prompt = (
            "Anda adalah AI financial assistant untuk aplikasi budgeting dan analytics.\n"
            "Jawab dalam Bahasa Indonesia.\n"
            "Jangan mengarang data yang tidak diberikan.\n"
            "Jika konteks kurang, jelaskan keterbatasannya secara singkat.\n\n"
            f"Ringkasan konteks sebelumnya:\n{context_summary or '-'}\n\n"
            f"Konteks finansial dari aplikasi:\n{finance_context or '-'}\n\n"
            f"Pertanyaan pengguna:\n{prompt}"
        )

        completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah AI financial assistant untuk aplikasi budgeting, transaksi, receipt OCR, "
                        "dan insight keuangan. Berikan jawaban yang ringkas, jelas, dan bisa ditindaklanjuti."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            model=self.settings.ai_model,
            temperature=0.2,
            timeout=30,
        )

        content = completion.choices[0].message.content
        if isinstance(content, list):
            answer = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        else:
            answer = str(content).strip()

        return AnalyzeResponse(
            answer=answer or "Data unavailable.",
            model=self.settings.ai_model,
            context_summary=context_summary or None,
        )

