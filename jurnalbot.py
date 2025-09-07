import os
import re
import dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from docx import Document 
from fpdf import FPDF
import unicodedata

# load environment variables from .env file
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.5, 
    api_key=GEMINI_API_KEY
)

# define a prompt template
prompt_template = """
Buat makalah ilmiah tentang topik berikut: "{topic}".
Gunakan format:
1. Pendahuluan
2. Tinjauan Pustaka
3. Metode
4. Hasil/Analisis
5. Kesimpulan

Tuliskan secara jelas, rapi, dan akademik.
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template
)

# create an LLM chain (baru style LangChain modern)
makalah_chain = prompt | llm

# tool
tool_makalah = Tool(
    name="Makalah Generator",
    func=lambda topic: makalah_chain.invoke({"topic": topic}).content,
    description="Digunakan untuk membuat makalah ilmiah tentang topik tertentu."
)

# agent prompt executor
agent_prompt = PromptTemplate.from_template("""
Kamu adalah asisten AI akademik yang bertugas membuat makalah ilmiah.
Kamu dapat menggunakan tool berikut untuk membantu menyelesaikan tugas:

{tools}

Aturan penggunaan:
- Jika perlu membuat makalah, pilih tool dari daftar {tool_names}.
- Gunakan format ReAct berikut:
  Thought: alasan pemilihan aksi
  Action: nama tool
  Action Input: input untuk tool
  Observation: hasil dari tool
  (ulangi langkah di atas jika perlu)
  Final Answer: jawaban akhir berupa makalah lengkap

User memberikan topik:
{input}

Riwayat interaksi sebelumnya:
{agent_scratchpad}

Ingat:
- Jawaban akhir HARUS ada di bagian "Final Answer".
- Tulis makalah ilmiah dengan struktur: Pendahuluan, Tinjauan Pustaka, Metode, Hasil/Analisis, Kesimpulan.
""")

# create an agent with the tool
agent = create_react_agent(llm, [tool_makalah], prompt=agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool_makalah],
    debug=True,
    handle_parsing_errors=True
)

# utilities
def save_to_docx(content, filename="makalah.docx"):
    doc = Document()
    for line in content.split('\n'):
        doc.add_paragraph(line)
    doc.save(filename)
    return filename

def get_font_path():
    """Cari font unicode di berbagai lokasi umum"""
    possible_fonts = [
        "C:/Windows/Fonts/arialuni.ttf",       
        "C:/Windows/Fonts/seguiemj.ttf",       
        "C:/Windows/Fonts/seguisym.ttf",       
        "C:/Windows/Fonts/arial.ttf",          
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  
    ]
    for path in possible_fonts:
        if os.path.exists(path):
            return path
    return None

def clean_text(text: str) -> str:
    """Bersihkan karakter aneh & markdown biar PDF lebih clean"""
    # hilangkan bold/italic Markdown (**...**, *...*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # hilangkan heading markdown (#, ##, ###)
    text = re.sub(r'#+\s', '', text)

    # normalisasi dash
    text = text.replace("—", "-").replace("–", "-")

    # normalisasi unicode → bentuk paling sederhana
    text = unicodedata.normalize("NFKD", text)

    # ganti kutip miring jadi kutip biasa
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

    # hapus karakter yang tidak bisa ditulis di PDF
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text.strip()

def save_to_pdf(content, filename="makalah.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # gunakan font built-in biar aman Unicode basic
    pdf.set_font("Helvetica", size=12)

    # hitung lebar halaman efektif
    page_width = pdf.w - 2 * pdf.l_margin

    for line in content.split('\n'):
        # multi_cell dengan lebar halaman efektif
        pdf.multi_cell(page_width, 10, line, align="J")
        pdf.ln(0.5)

    pdf.output(filename)
    return filename

def save_to_md(content, filename="makalah.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

# streamlit UI
st.title("AI Docs Agent - Makalah Generator")
st.write("Masukkan topik untuk membuat makalah ilmiah.")
topic = st.text_input("Topik Makalah", "Dampak Perubahan Iklim terhadap Keanekaragaman Hayati")

if st.button("Buat Makalah"):
    with st.spinner("Membuat makalah..."):
        try:
            response = agent_executor.invoke({"input": topic})
            response = response["output"]

            st.session_state["makalah"] = response  # simpan hasil di session state

            st.success("Makalah berhasil dibuat!")
            st.text_area("Hasil Makalah", response, height=300)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# bagian download muncul kalau makalah sudah ada
if "makalah" in st.session_state:
    makalah = st.session_state["makalah"]

    docx_file = save_to_docx(clean_text(makalah))
    pdf_file = save_to_pdf(clean_text(makalah))
    md_file = save_to_md(makalah)

    # pilihan format download
    option = st.selectbox(
        "Pilih format file untuk diunduh:",
        ("DOCX", "PDF", "Markdown (.md)")
    )

    # mapping pilihan ke file
    file_map = {
        "DOCX": (docx_file, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "PDF": (pdf_file, "application/pdf"),
        "Markdown (.md)": (md_file, "text/markdown"),
    }

    file_name, mime_type = file_map[option]

    with open(file_name, "rb") as f:
        file_bytes = f.read()

    # tombol download
    st.download_button(
        label=f"Download {option}",
        data=file_bytes,
        file_name=file_name,
        mime=mime_type
    )

st.write("Developed by Naufal Devops")
