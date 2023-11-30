# ai-assistant

## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name assistant -f environment.yaml --force
```

Activate the environment.
```bash
conda activate assistant
```

Download and save the LLM artefacts in `models/`. The models used in this demo are downloaded from [TheBloke](https://huggingface.co/TheBloke).
- [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)
- [CodeLlama-7B-GGUF](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/tree/main)

We shall also use Google PaLM, SERP API, News API and Wolfram Alpha. As such, the following API keys are required:
- PaLM: `GOOGLE_API_KEY`
- SERP API: `SERPAPI_API_KEY`
- News API: `NEWSAPI_API_KEY`
- Wolfram Alpha: `WOLFRAM_ALPHA_APPID`
Save these keys in `.env`.


## 💻 App

We use Streamlit as the interface for the demos:
```bash
streamlit run app.py
```
