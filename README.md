# GenAI Notebooks
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/shahabas9/GenAI)

This repository contains a collection of Jupyter notebooks dedicated to exploring various concepts and applications in Generative AI. It serves as a practical guide and learning resource for fine-tuning large language models, building Retrieval-Augmented Generation (RAG) systems, and working with multimodal data.

The notebooks provide hands-on examples using a variety of state-of-the-art models, frameworks, and tools.

## Key Areas Covered

The projects in this repository are broadly categorized into the following areas:

### 1. Model Fine-Tuning

Explore various Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt large language models for specific tasks without the need for extensive computational resources.

*   **`Fine_tune_Llama_2.ipynb`**: A comprehensive walkthrough of fine-tuning Llama 2 using 4-bit quantization with QLoRA, `bitsandbytes`, and the TRL library.
*   **`lora_tuning.ipynb`**: Demonstrates how to apply Low-Rank Adaptation (LoRA) to Gemma models using KerasNLP.
*   **`Fine_Tuning_with_Mistral_QLora_PEFt.ipynb`**: A notebook for fine-tuning Mistral models with QLoRA and PEFT.

### 2. Retrieval-Augmented Generation (RAG)

Learn to build RAG systems that enhance LLM responses by retrieving relevant information from external knowledge bases.

*   **`RAG_from_scratch.ipynb`**: Implements a basic RAG pipeline from the ground up to explain the fundamental concepts of document retrieval and context-based generation.
*   **`RAG_with_langhain_faiss_gemini.ipynb`**: A complete RAG application using LangChain, Google Gemini, and the FAISS vector store for efficient similarity search.
*   **`RAG_Application_Using_Haystack_and_OpenAI.ipynb`**: Builds a search and question-answering system using the Haystack framework, leveraging document stores and OpenAI models.
*   **`RAG_with_Mistral_LAngChain_Weaviate.ipynb`**: An implementation of a RAG pipeline using Mistral, LangChain, and the Weaviate vector database.
*   **`RAG_APP_with_huggingface_google_gemma_and_MongoDB.ipynb`**: Demonstrates a RAG pipeline integrating Google's Gemma model with MongoDB as the data source.

### 3. Multimodal RAG

Dive into advanced RAG systems that can process and reason over multiple data types, including text, images, and tables. All related notebooks are in the `multi-model/` directory.

*   **`Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt.ipynb`**: A practical guide to building a multimodal RAG system with Gemini and LangChain, capable of understanding both text and images.
*   **`Multimodal_RAG_Using_Google_Gemini_llamaindex.ipynb`**: Explores creating multimodal search applications using Gemini and LlamaIndex.
*   **`Extract_Image,Table,Text_from_Document_MultiModal_Summrizer_RAG_App.ipynb`**: A project focused on extracting different content types from documents to build a multimodal summarizer.

### 4. Vector Databases and Integrations

Notebooks demonstrating the integration of various vector databases and traditional databases in GenAI workflows.

*   **`MongoDB_with_pinecone_part1.ipynb`** & **`MongoDB_with_pinecone_part2.ipynb`**: Shows how to use MongoDB in conjunction with Pinecone for scalable vector search applications.
*   Integrations with **FAISS**, **Weaviate**, and **LanceDB** are also showcased across different RAG notebooks.

## Core Technologies

This repository utilizes a wide range of modern AI technologies:

*   **LLMs**: Google Gemini, Llama 2, Mistral, Google Gemma, OpenAI models.
*   **Frameworks**: LangChain, KerasNLP, LlamaIndex, Haystack, Transformers, PEFT.
*   **Vector Databases**: FAISS, Weaviate, LanceDB, Pinecone.
*   **Tools**: `bitsandbytes`, `trl`, Google AI Studio.

## Getting Started

1.  Clone the repository to your local machine:
    ```sh
    git clone https://github.com/shahabas9/GenAI.git
    ```
2.  Navigate to the repository directory:
    ```sh
    cd GenAI
    ```
3.  Open the Jupyter notebooks using your preferred environment (like Jupyter Lab, VS Code, or Google Colab).
4.  Install the required dependencies. Most notebooks include `pip install` commands in the first few cells for easy setup.
5.  Ensure you have the necessary API keys (e.g., for Google AI, OpenAI, or Kaggle) and configure them as shown in the notebooks, often using environment variables or user data secrets in Colab.