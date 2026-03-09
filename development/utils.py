from pypdf import PdfReader
import io

def extract_text_from_pdf(file_storage):
    """
    Menerima FileStorage dari Flask, mengembalikan string teks.
    """
    try:
        # Membaca file langsung dari memory tanpa save ke disk dulu
        reader = PdfReader(file_storage)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None