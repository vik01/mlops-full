FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

# Step 1: copy environment files first (for caching)
COPY environment.yml .
COPY requirements.txt .

# Step 2: create conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Step 3: copy project code
COPY . .

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "Heart_Disease_Prediction", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]