import streamlit as st
import requests

# Set the FastAPI backend URL.
# Adjust the port and host as needed.
API_URL = "http://localhost:8000"

def upload_files(files):
    upload_endpoint = f"{API_URL}/upload/"
    # Prepare file tuples for multiple file upload:
    file_tuples = []
    for file in files:
        # Each file tuple consists of (form_field_name, (filename, file_object, content_type))
        file_tuples.append(("files", (file.name, file, "application/pdf")))
    response = requests.post(upload_endpoint, files=file_tuples)
    if response.ok:
        return response.json()
    else:
        st.error(f"Upload error: {response.text}")
        return None

def query_documents(query):
    query_endpoint = f"{API_URL}/query/"
    # Send query as form data.
    response = requests.post(query_endpoint, data={"query": query})
    if response.ok:
        return response.json()
    else:
        st.error(f"Query error: {response.text}")
        return None

def delete_all_data():
    delete_endpoint = f"{API_URL}/delete/"
    response = requests.delete(delete_endpoint)
    if response.ok:
        return response.json()
    else:
        st.error(f"Delete error: {response.text}")
        return None

st.title("PDF Embedder & Query Interface")

# Section 1: Uploading PDF files
st.header("Upload PDF Files")
uploaded_files = st.file_uploader("Choose PDF files to upload", type=["pdf"], accept_multiple_files=True)
if st.button("Upload Files"):
    if not uploaded_files:
        st.error("Please select at least one PDF file to upload.")
    else:
        with st.spinner("Uploading..."):
            result = upload_files(uploaded_files)
        if result:
            st.success(result.get("message", "Upload successful."))
            st.write("Uploaded Files:", result.get("files", []))

# Section 2: Query Documents
st.header("Query Documents")
query_input = st.text_input("Enter your query:")
if st.button("Submit Query"):
    if not query_input:
        st.error("Please enter a query.")
    else:
        with st.spinner("Querying..."):
            result = query_documents(query_input)
        if result:
            st.write("Response:", result.get("response"))
            st.write("Latency (s):", result.get("latency_seconds"))

# Section 3: Delete Data
st.header("Delete All Data")
if st.button("Delete Data"):
    confirm_delete = st.checkbox("I confirm that I want to delete all documents and the ChromaDB.")
    if confirm_delete:
        with st.spinner("Deleting..."):
            result = delete_all_data()
        if result:
            st.success(result.get("message"))
    else:
        st.warning("Please confirm deletion by checking the box above.")
