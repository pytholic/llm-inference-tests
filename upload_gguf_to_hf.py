from huggingface_hub import create_repo, HfApi
hf_token = "" # Specify your token
username = "" # Specify your username
api = HfApi()

MODEL_NAME = "Phi-3.5-mini-instruct-GGUF"

# Create empty repo
create_repo(
    repo_id = f"{username}/{MODEL_NAME}",
    repo_type="model",
    exist_ok=True,
    token=hf_token
)
# Upload gguf files
api.upload_folder(
    folder_path=f"models_gguf/{MODEL_NAME}",
    repo_id=f"{username}/{MODEL_NAME}",
    allow_patterns=f"*.gguf",
    token=hf_token
)
