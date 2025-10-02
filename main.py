import os
import re
import io
import dotenv
import uvicorn
import unicodedata
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from docx import Document
from fpdf import FPDF

# load environment variables from .env file
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0.5, 
    api_key=GEMINI_API_KEY
)

# initialisasi fastAPi
app = FastAPI(title="AI Docs Agent", description="AI Agent untuk Generator Makalah")

# CORS middleware for connect backend FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# template prompt untuk AI Agent
prompt_template = """
Buat makalah ilmiah tentang topik berikut: "{topic}".

**PERATURAN PENTING:**
- **Langsung mulai dengan judul makalah.** Jangan sertakan kalimat pembuka atau basa-basi seperti "Tentu, berikut adalah makalah..."
- **Jangan gunakan pemisah** seperti "---" atau baris kosong yang berlebihan.

Gunakan format terstruktur berikut:
1. Pendahuluan
2. Tinjauan Pustaka
3. Metode Penelitian
4. Hasil dan Analisis
5. Kesimpulan
6. Daftar Pustaka

Tuliskan secara jelas, terstruktur, dan dengan gaya bahasa akademik.
"""

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names"],
    template=prompt_template,
)

# chain LangChain
makalah_chain = prompt | llm

# tool untuk membuat makalah
tool_makalah = Tool(
    name="Makalah Generator",
    func=lambda topic: makalah_chain.invoke({"topic": topic}).content,
    description="Digunakan untuk membuat makalah ilmiah tentang topik tertentu."
)

# prompt untuk react agent
agent_prompt = PromptTemplate.from_template("""
Kamu adalah asisten AI akademik yang bertugas membuat makalah ilmiah.
Kamu dapat menggunakan tool berikut untuk membantu menyelesaikan tugas:

{tools}

Aturan penggunaan:
- Jika perlu membuat makalah, pilih tool dari daftar {tool_names}.
- Gunakan format ReAct berikut:
Thought: Alasan pemilihan aksi.
Action: Nama tool yang digunakan.
Action Input: Input untuk tool tersebut.
Observation: Hasil dari eksekusi tool.
(Ulangi pola Thought/Action/Action Input/Observation jika diperlukan)
Thought: Saya sudah memiliki semua informasi yang dibutuhkan.
Final Answer: Jawaban akhir berupa makalah ilmiah yang lengkap dan terstruktur.

User memberikan topik:
{input}

Riwayat interaksi sebelumnya:
{agent_scratchpad}

Ingat: Jawaban akhir HARUS berada di dalam "Final Answer".
""")

# agent untuk menjalankan tool
agent = create_react_agent(llm, [tool_makalah], prompt=agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool_makalah],
    debug=True,
    handle_parsing_errors=True
)

# function untuk utilitas file
def slugify(value: str) -> str:
    """
    Mengubah string menjadi format yang aman untuk nama file.
    Contoh: "Dampak Perubahan Iklim!" -> "dampak_perubahan_iklim"
    """
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '_', value)
    return value

def clean_text(text: str) -> str:
    """Membersihkan teks dari markdown dan karakter aneh untuk ekspor file."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s', '', text)
    text = text.replace("—", "-").replace("–", "-")
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    cleaned_text = text.encode('latin-1', 'replace').decode('latin-1')
    return cleaned_text.strip()

def save_to_docx_stream(content):
    """Menyimpan konten ke DOCX dalam bentuk stream in-memory."""
    doc = Document()
    # FIX: Membersihkan konten sebelum menambahkannya ke DOCX
    cleaned_content = clean_text(content)
    for line in cleaned_content.split('\n'):
        doc.add_paragraph(line)
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    return stream

def save_to_pdf_stream(content):
    """Menyimpan konten ke PDF dalam bentuk stream in-memory."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    cleaned_content = clean_text(content)
    pdf.write(8, cleaned_content)
    # FIX: Mengubah .decode() menjadi .encode() untuk menghasilkan objek bytes
    pdf_bytes = pdf.output()
    return io.BytesIO(pdf_bytes)

def save_to_md_stream(content):
    """Menyimpan konten ke Markdown dalam bentuk stream in-memory."""
    # Konten markdown tidak perlu dibersihkan
    return io.BytesIO(content.encode('utf-8'))

# model pydantic untuk validasi content & response
class TopicRequest(BaseModel):
    topic: str

class MakalahResponse(BaseModel):
    content: str

class DownloadRequest(BaseModel):
    content: str
    format: str
    title: str

# endpoint fastAPI
@app.get("/", response_class=FileResponse)
async def read_root():
    """Endpoint untuk menyajikan file index.html sebagai halaman utama."""
    # Pastikan file index.html berada di folder yang sama dengan main.py
    return "index.html"

@app.post("/generate-makalah", response_model=MakalahResponse)
async def generate_makalah(request: TopicRequest):
    """Endpoint untuk menghasilkan konten makalah."""
    try:
        response = await agent_executor.ainvoke({"input": request.topic})
        return MakalahResponse(content=response["output"])
    except Exception as e:
        return MakalahResponse(content=f"Terjadi kesalahan: {str(e)}")

@app.post("/download")
async def download_file(request: DownloadRequest):
    """Endpoint untuk mengunduh makalah dalam format yang dipilih."""
    format_map = {
        "docx": {"function": save_to_docx_stream, "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "pdf": {"function": save_to_pdf_stream, "mime": "application/pdf"},
        "md": {"function": save_to_md_stream, "mime": "text/markdown"}
    }
    selected_format = format_map.get(request.format)
    if not selected_format:
        return {"error": "Format tidak valid."}
        
    stream = selected_format["function"](request.content)
    
    # Membuat nama file dinamis dari judul
    safe_title = slugify(request.title) if request.title else "makalah"
    filename = f"{safe_title}.{request.format}"
    
    headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
    return StreamingResponse(stream, media_type=selected_format["mime"], headers=headers)

# run app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)




