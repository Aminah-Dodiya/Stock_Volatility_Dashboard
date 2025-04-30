import os
from pydantic import BaseSettings

def return_full_path(filename: str = ".env") -> str:
    """Return the full path of the .env file."""
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    return os.path.join(directory_name, filename)

class Settings(BaseSettings):
    """Settings loaded from .env file."""
    alpha_api_key: str
    db_name: str
    model_directory: str

    class Config:
        env_file = return_full_path(".env")

settings = Settings()
