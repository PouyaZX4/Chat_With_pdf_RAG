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

git clone https://github.com/your-username/your-repo-name.git
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
main.py: The main Streamlit application script. It handles PDF uploading, text processing, LLM loading, LangChain integration, and the overall UI flow.

htmlTemplate.py: Contains custom HTML and CSS templates for styling the chat messages within the Streamlit interface, providing a more engaging user experience.

🧠 Technical Details
Language Model: Qwen/Qwen2-1.5B from Hugging Face, loaded via AutoModelForCausalLM and AutoTokenizer.

Text Processing: PyPDF2 for reading PDF content and RecursiveCharacterTextSplitter from LangChain for chunking text.

Embeddings: HuggingFaceEmbeddings with all-MiniLM-L6-v2 model for converting text chunks into vector representations.

Vector Store: Chroma is used to store and retrieve document embeddings efficiently.

Retrieval-Augmented Generation (RAG): LangChain's PromptTemplate, RunnablePassthrough, and RunnableLambda are used to construct a robust Q&A chain that retrieves relevant document chunks before generating a response.

Frontend: Streamlit provides the interactive web interface.

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

by [Pouya/PouyaZX4]
