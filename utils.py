from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# PDF Okuma ve Chunk Oluşturma
def read_pdf_and_create_chunks(pdf_path, chunk_size=500, chunk_overlap=50):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# PDF'den Chroma Veritabanına Ekleme
def add_pdf_to_chroma_db(pdf_path, chroma_db, chunk_size=200, chunk_overlap=50):
    chunks = read_pdf_and_create_chunks(pdf_path, chunk_size, chunk_overlap)
    chroma_db.add_texts(texts=chunks)
    print(f"{len(chunks)} chunk başarıyla Chroma veritabanına eklendi.")

# Embedding Modeli
def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")

# Chroma Veritabanı
def load_chroma_db(persist_directory="chroma_index"):
    embedding_model = create_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

# LLM Modeli
def load_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.4,
        max_tokens=524
    )

# Prompt Template Tanımı
def create_prompt_template():
    prompt_template = (
        """
        İnsan: Aşağıdaki bağlamı kullanarak soruya detaylı bir şekilde ve anlamlı bir şekilde yanıt ver. Eğer cevabı bilmiyorsan, bunu belirt ve uydurma. 
        Cevaplarını sadece sana verilen bilgiler doğrultusunda ver. Sana verilen veri dışına çıkmadan yanıtlaman lazım.

        <bağlam>
        {context}
        </bağlam>

        Soru: {question}

        Asistan: """
    )
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA Zinciri Tanımı
def create_qa_chain(chroma_db, llm, prompt_template):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

# Yanıt Oluşturma
def generate_response_with_context(query, qa, chroma_db):
    retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)

    context = " ".join([doc.page_content for doc in relevant_docs])

    # QA zinciri ile yanıt oluştur
    response = qa.invoke({"context": context, "question": query})

    return {
        "response": response.get("result", "Yanıt bulunamadı."),
        "context_used": context,
        "source_documents": [doc.page_content for doc in relevant_docs]
    }

# Ana Akış
if __name__ == "__main__":
    # Dosya ve veritabanı yükleme
    pdf_path = "/Users/ayse/PycharmProjects/Chatbot/source_doc/newmerge.pdf"
    persist_directory = "/Users/ayse/PycharmProjects/Chatbot/chroma_index"

    chroma_db = load_chroma_db(persist_directory)
    add_pdf_to_chroma_db(pdf_path, chroma_db)

    llm = load_llm()
    prompt_template = create_prompt_template()
    qa = create_qa_chain(chroma_db, llm, prompt_template)

    # Soru sorma
    query = "Okul kartımı kaybettim, ne yapabilirim?"
    response = generate_response_with_context(query, qa, chroma_db)

    # Yanıtı yazdır
    print("\nYanıt:")
    print(response["response"])

    print("\nKullanılan Bağlam:")
    print(response["context_used"])

    print("\nKaynak Belgeler:")
    for doc in response["source_documents"]:
        print(doc)