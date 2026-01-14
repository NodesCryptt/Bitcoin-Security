FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api.py .

# Create data directory
RUN mkdir -p data

# Expose port
EXPOSE 5000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
