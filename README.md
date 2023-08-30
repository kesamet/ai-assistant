# ai-assistant

The app use quantized llama-2 models and run on CPU with acceptable inference time (~1 min).


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name assistant -f environment.yaml --force
```

Activate the environment.
```bash
conda activate assistant
```

Download the LLM artefact. The model used in this demo is downloaded from [TheBloke](https://huggingface.co/TheBloke).
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin -P ./models
```
```bash
wget https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/blob/main/codellama-7b.Q2_K.gguf -P ./models
```

You can also use another model of your choice. Ensure that it can be loaded by `langchain.llms.CTransformers` and update `config.yaml`.


## 💻 App

We use Streamlit as the interface for the demos. There are two demos:
- Retrieval QA
```bash
streamlit run app.py
```
