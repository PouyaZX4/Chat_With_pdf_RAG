PDF Chatbot with Hugging Face LLM
📚 Project Overview
This project presents an interactive Streamlit application that allows users to upload PDF documents and engage in a conversational Q&A session with their content. Leveraging the power of Hugging Face's Qwen/Qwen2-1.5B language model and LangChain for efficient document processing and retrieval, this tool transforms static PDFs into dynamic knowledge bases.

Say goodbye to manual searching through lengthy documents! Simply upload your PDFs, and ask questions directly to your content.

✨ Features
PDF Upload: Easily upload one or multiple PDF documents.

Intelligent Q&A: Ask natural language questions about the content of your uploaded PDFs.

Hugging Face LLM Integration: Powered by the Qwen/Qwen2-1.5B model for robust language understanding and generation.

LangChain Integration: Utilizes LangChain for text splitting, embedding, and efficient retrieval-augmented generation (RAG).

User-Friendly Interface: Built with Streamlit for a clean and intuitive web interface.

Responsive Chat UI: Custom HTML/CSS templates provide a visually appealing chat experience.

🚀 Getting Started
Follow these steps to get your PDF Chatbot up and running on your local machine.

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.9 or higher

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/PouyaZX4/PDF_Chatbot_with_HF_LLM.git
cd your-repo-name

(Remember to replace your-username/your-repo-name with your actual GitHub repository details.)

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

(You'll need to create a requirements.txt file. See the next section.)

requirements.txt
Create a file named requirements.txt in the root of your project and add the following content:

streamlit
PyPDF2
transformers
langchain
langchain-huggingface
langchain-community
sentence-transformers
torch

Running the Application
Ensure your virtual environment is active.

Run the Streamlit application:

streamlit run main.py

Open your web browser and navigate to the URL displayed in your terminal (usually http://localhost:8501).

💡 How to Use
Upload PDFs: On the left sidebar, click the "Upload PDF(s)" button and select the PDF files you want to chat with.

Process Documents: After uploading, click the "Process" button in the sidebar. The application will then process the text from your PDFs and prepare it for Q&A. This might take a few moments depending on the size and number of documents.

Ask Questions: Once processing is complete, a text input field will appear. Type your question related to the content of your PDFs and press Enter.

Get Answers: The AI chatbot will respond with an answer based only on the information found within your uploaded documents. If it cannot find an answer in the provided context, it will state "I don’t know."

⚙️ Project Structure
main.py: The core Streamlit application script. It orchestrates PDF handling, LLM interaction, and the user interface.

htmlTemplate.py: Defines the visual styling for the chat messages, ensuring a consistent and appealing look for user and bot interactions.

🧠 Technical Details
This section dives a bit deeper into the technical implementation, explaining how different components work together in main.py and htmlTemplate.py.

Language Model (Qwen/Qwen2-1.5B):

Implementation: In main.py, the load_model() function is responsible for downloading and initializing the Qwen/Qwen2-1.5B model and its tokenizer from Hugging Face. It then wraps this into a HuggingFacePipeline for easy integration with LangChain. This model is the brain of the chatbot, responsible for generating human-like responses.

Text Processing (PyPDF2 & RecursiveCharacterTextSplitter):

Implementation: When PDFs are uploaded in main.py, PyPDF2.PdfReader is used to extract raw text from each page. This raw text is then passed to get_text_chunks(), which employs LangChain's RecursiveCharacterTextSplitter. This splitter breaks down the large text into smaller, manageable "chunks" (of 500 characters with 200 characters overlap) to optimize embedding and retrieval processes.

Embeddings (HuggingFaceEmbeddings with all-MiniLM-L6-v2):

Implementation: The get_vectorstore() function in main.py takes these text chunks and converts them into numerical vector representations using HuggingFaceEmbeddings with the all-MiniLM-L6-v2 model. These embeddings capture the semantic meaning of the text, allowing for efficient similarity searches.

Vector Store (Chroma):

Implementation: Once the text chunks are embedded, Chroma.from_texts() is used in get_vectorstore() to create a Chroma vector store. This database stores the text chunks alongside their embeddings, making it quick to retrieve relevant information based on a user's query.

Retrieval-Augmented Generation (RAG) Chain:

Implementation: The get_qa_chain() function in main.py constructs the core RAG pipeline using LangChain.

It defines a PromptTemplate that guides the LLM to answer questions only based on the provided textbook_context.

The retriever (created from the Chroma vector store) fetches the most relevant text chunks (context) based on the user's question.

RunnablePassthrough and RunnableLambda are used to pass the question, retrieved context, and (optionally) student profile and conversation history to the prompt.

Finally, the llm processes the prompt and context, and StrOutputParser() extracts the generated answer as a string. This entire chain ensures that the LLM's responses are grounded in the uploaded PDF content.

Frontend (Streamlit & htmlTemplate.py):

Implementation: main.py uses Streamlit's st.set_page_config, st.sidebar.file_uploader, st.sidebar.button, and st.text_input to create the interactive web interface.

htmlTemplate.py provides the css, user_template, and bot_template strings. These are injected into the Streamlit app using st.markdown(..., unsafe_allow_html=True) to give the chat messages a custom, visually appealing look with avatars and distinct styling for user and bot messages.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📄 License
This project is open-source and available under the MIT License.

Made with ❤️ by [Pouya/PouyaZX4]
