import os

# API Configuration
API_PORT = int(os.getenv("PORT", 13675))

# Path Configuration
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "out")
LOG_DIR = os.getenv("LOG_DIR", None)

# Garbage Collector Configuration
GC_MAX_AGE_HOURS = int(os.getenv("GC_MAX_AGE_HOURS", "24"))
GC_INTERVAL_MINUTES = int(os.getenv("GC_INTERVAL_MINUTES", "10"))
