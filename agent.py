import re
import json
import os
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
   

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- Text Splitter (Custom Implementation) --------------------
def simple_text_splitter(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    A simple text splitter that splits text into chunks with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# -------------------- NEW: PDF Processing and Vector Store --------------------
class InMemoryVectorStore:
    def __init__(self):
        self.embeddings_model = "text-embedding-004"
        self.store = [] # List of (text_chunk, embedding)

    def add_texts(self, texts: list[str]):
        print("[VectorStore] Generating embeddings for text chunks...")
        for text in texts:
            response = genai.embed_content(model=self.embeddings_model, content=text)
            embedding = response['embedding']
            self.store.append((text, np.array(embedding)))
        print(f"[VectorStore] Added {len(texts)} chunks to the store.")

    def similarity_search(self, query: str, k: int = 4) -> list[str]:
        if not self.store:
            return []

        query_embedding_response = genai.embed_content(model=self.embeddings_model, content=query)
        query_embedding = np.array(query_embedding_response['embedding'])

        similarities = []
        for i, (text, embedding) in enumerate(self.store):
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((sim, text))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [text for sim, text in similarities[:k]]


def create_vector_store_from_pdf(pdf_path: str, google_api_key: str) -> InMemoryVectorStore:
    """
    Reads a PDF, splits the text into chunks, creates embeddings using Google Gemini,
    and stores them in a custom in-memory vector store.
    """
    print("[Manager] Creating vector store from PDF:", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)

    # Using custom simple text splitter
    chunks = simple_text_splitter(text, chunk_size=1000, chunk_overlap=200)

    vector_store = InMemoryVectorStore()
    vector_store.add_texts(chunks)

    print("[Manager] Vector store created successfully.")
    return vector_store


# -------------------- SUB AGENT TOOLS --------------------
def call_po_api(po_number: str):
    """
    Example API call using PO number. Replace this with your actual endpoint.
    """
    print("[SubAgent] Calling API with PO number:", po_number)
    # url = "https://my-api.com/get-po" 
    response = {"po_no":1456768,
                "milestone":[{"item 1":{"price":250000, "quantity":50}},{"item 2":{"price":150000, "quantity":50}}, {"item 3":{"price":200000, "quantity":50}} ]
                }

    # if response.status_code != 200:
    #     raise Exception(f"API call failed: {response.status_code}")

    print("[SubAgent] API call successful.")
    print(response)
    return response


def llm_validation(model, invoice_data: dict, po_data: dict):
    print("[SubAgent] Performing LLM-based validation...")
    prompt = f"""
    You are an expert invoice validation agent. Your task is to compare the extracted invoice data with the purchase order (PO) data and identify any discrepancies if the quanity is lee than or equal to listed quanity we can consider it valid.

    **Invoice Data:**
    ```json
    {json.dumps(invoice_data, indent=2)}
    ```

    **Purchase Order (PO) Data:**
    ```json
    {json.dumps(po_data, indent=2)}
    ```

    **Instructions:**
    1.  Compare the items listed in the invoice with the items in the PO.
    2.  For each item, verify that the quantity and price on the invoice match the PO.
    3.  Provide a summary of your findings. If there are discrepancies, list them clearly. If everything matches, confirm that the invoice is valid.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[SubAgent] LLM validation failed: {e}")
        return "LLM validation could not be performed."


def validate_invoice_data(model, invoice_data: dict, po_data: dict):
    """
    Performs LLM-based validation of the invoice against the PO data.
    """
    print("[SubAgent] Performing LLM-based validation...")

    llm_summary = llm_validation(model, invoice_data, po_data)

    return {
        "invoice_number": invoice_data.get("invoice_number", "N/A"),
        "po_number": invoice_data.get("po_number", "N/A"),
        "llm_validation_summary": llm_summary
    }


# -------------------- MANAGER AGENT (Modified for Generic Google GenAI) --------------------
class ManagerAgent:
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name)

    def query_gemini(self, context: str, question: str):
        """
        Queries the Gemini model with context and a question.
        """
        prompt = f"""Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def process_invoice(self, invoice_path: str, google_api_key: str):
        print("[Manager] Starting invoice processing with embedding and Google Gemini (generic)...")

        vector_store = create_vector_store_from_pdf(invoice_path, google_api_key)
    
        # Query for the required information
        po_number_query = "What is the value for PO No, or purchange order number or Po number"
        invoice_date_query = "What is the invoice date?"
        vendor_query = "What is the vendor's name?"
        items_query = "list all the items with their descriptions, per unit cost, quantity and amounts as a JSON object"
        invoice_number_query = "what is the invoice number listed in document"

        # Retrieve relevant chunks for each query
        po_number_context = "\n".join(vector_store.similarity_search(po_number_query))
        invoice_date_context = "\n".join(vector_store.similarity_search(invoice_date_query))
        vendor_context = "\n".join(vector_store.similarity_search(vendor_query))
        items_context = "\n".join(vector_store.similarity_search(items_query))
        invoice_number_context = "\n".join(vector_store.similarity_search(invoice_number_query))

        po_number = self.query_gemini(po_number_context, po_number_query)
        invoice_date = self.query_gemini(invoice_date_context, invoice_date_query)
        vendor = self.query_gemini(vendor_context, vendor_query)
        items_json_str = self.query_gemini(items_context, items_query)
        invoice_number = self.query_gemini(invoice_number_context, invoice_number_query)

        try:
            print(type(items_json_str))
            clean = items_json_str.encode('utf-8').decode('unicode_escape')
            clean = items_json_str.strip("`").strip()
            clean = clean[clean.find("{"):]
            print(clean)
            items = json.loads(clean)
        except json.JSONDecodeError:
            print("[Manager] Error: Could not parse items from the response. Raw response:", items_json_str)
            items = []

        invoice_data = {
            "po_number": po_number,
            "invoice_date": invoice_date,
            "vendor": vendor,
            "items": items,
            "invoice_number": invoice_number
        }

        print("[Manager] Extracted invoice data:", invoice_data)

        if not po_number:
            raise ValueError("PO number not found in invoice.")

        po_data = call_po_api(po_number)
        validation_report = validate_invoice_data(self.model, invoice_data, po_data)

        print("\n✅ Validation Completed. Summary:")
        print(json.dumps(validation_report, indent=2))

        return validation_report


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
    print(f"Using Google Gemini model: {GEMINI_MODEL_NAME}")


    invoice_file = "INVOICe.pdf"
    manager = ManagerAgent(
        model_name=GEMINI_MODEL_NAME
    )

    try:
        result = manager.process_invoice(invoice_file, google_api_key=GOOGLE_API_KEY)
        print("\n✅ Final Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print("❌ Error:", e)
