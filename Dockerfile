FROM python:3.12-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "[IP_ADDRESS]", "--port", "8000"]