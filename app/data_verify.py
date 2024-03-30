from langchain_community.document_loaders import DirectoryLoader

def load_documents():
    xlsx_loader = DirectoryLoader(r'C:/Users/PC/Downloads/Long 4_7_2024/AI_lord/data', glob="**/*.xlsx")
    xlsx_docs = xlsx_loader.load()

    # DirectoryLoader cho file .txt
    text_loader = DirectoryLoader(r'C:/Users/PC/Downloads/Long 4_7_2024/AI_lord/data', glob="**/*.txt")
    text_docs = text_loader.load()

    # Kết hợp dữ liệu từ cả hai loại loader
    docs = text_docs + xlsx_docs

    return docs