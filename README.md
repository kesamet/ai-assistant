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


## 💻 Vision Assistant App

Download `ggml-model` and `mmprog-model` from [mys/ggml_llava-v1.5-7b](https://huggingface.co/mys/ggml_llava-v1.5-7b) and save them in `models/llava-7b/`. Update `CLIP_MODEL_PATH` and `LLAVA_MODEL_PATH` in `config.yaml` accordingly.

Deploy LLAvA model as an endpoint.
```bash
python -m serve_llava
```

Run Streamlit app and select `Vision Assistant`.
```bash
streamlit run app.py
```

<p align="center">
    !<img src="./assets/screenshot.png" alt="screenshot" width="1000"/>
</p>


## 💻 AI Assistant App

We shall also use Google PaLM, SERP API, News API and Wolfram Alpha. As such, the following API keys are required:
- PaLM: `GOOGLE_API_KEY`
- SERP API: `SERPAPI_API_KEY`
- News API: `NEWSAPI_API_KEY`
- Wolfram Alpha: `WOLFRAM_ALPHA_APPID`
Save these keys in `.env`.

Run Streamlit app and select `AI Assistant`.
```bash
streamlit run app.py
```
