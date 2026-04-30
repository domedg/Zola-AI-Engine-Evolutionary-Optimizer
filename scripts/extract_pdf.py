import sys
import json

def extract_pdf(pdf_path, out_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            for page in doc:
                f.write(page.get_text())
        print(f"Success with PyMuPDF for {pdf_path}")
        return True
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            with open(out_path, 'w', encoding='utf-8') as f:
                for page in reader.pages:
                    f.write(page.extract_text())
            print(f"Success with pypdf for {pdf_path}")
            return True
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                with open(out_path, 'w', encoding='utf-8') as f:
                    for page in reader.pages:
                        f.write(page.extract_text())
                print(f"Success with PyPDF2 for {pdf_path}")
                return True
            except ImportError:
                print("No PDF library available. Please run: pip install pypdf")
                return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if extract_pdf(sys.argv[1], sys.argv[2]):
        sys.exit(0)
    else:
        sys.exit(1)
