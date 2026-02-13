import logging
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Equip guardian Angel"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Machine Learning API"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = ["*"]  # Specify type as list[str]
    
    # Logging
    LOG_LEVEL: int = logging.INFO

settings = Settings()

# Configure Global Logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(settings.PROJECT_NAME)
