# Roman to Devanagari Transliteration using Seq2Seq Modelling

This project aims to develop a transliteration system that converts Roman script to Devanagari script using a Sequence-to-Sequence (Seq2Seq) model. The project incorporates MLOps for model tracking and hyperparameter optimization, utilizes FastAPI to serve the model, and employs Prometheus and Grafana for monitoring and visualization of metrics.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
4. [API Endpoints](#api-endpoints)
5. [Monitoring](#monitoring)
6. [Configuration Files](#configuration-files)
7. [Acknowledgements](#acknowledgements)

## Project Structure

```
.
├── api.py
├── build_vocab.py
├── docker-compose.yml
├── Dockerfile
├── idx_to_target_char.json
├── input_vocab.json
├── prometheus.yml
├── requirements.txt
├── target_vocab.json
└── transliteration_params_sweep.py
```

### File Descriptions
- **api.py**: FastAPI application to serve the transliteration model.
- **build_vocab.py**: Script to build vocabulary for Roman and Devanagari characters.
- **docker-compose.yml**: Docker Compose configuration for setting up the application stack.
- **Dockerfile**: Docker configuration for building the application image.
- **idx_to_target_char.json**: Mapping from indices to Devanagari characters.
- **input_vocab.json**: Vocabulary of input Roman characters.
- **prometheus.yml**: Prometheus configuration for scraping metrics.
- **requirements.txt**: List of dependencies required for the project.
- **target_vocab.json**: Vocabulary of target Devanagari characters.
- **transliteration_params_sweep.py**: Script for hyperparameter optimization using sweeps.

## Setup and Installation

### Prerequisites
- Docker
- Docker Compose

### Installation Steps
1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/roman-to-devanagari-transliteration.git
   cd roman-to-devanagari-transliteration
   ```

2. **Build Docker Image**
   ```sh
   docker-compose build
   ```

3. **Start the Application**
   ```sh
   docker-compose up
   ```

4. **Install Python Dependencies (for local development)**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Transliteration Model
After starting the application using Docker Compose, the FastAPI server will be running and can be accessed at `http://localhost:8000`.

### Transliteration via API
You can use the `/transliterate` endpoint to get the Devanagari transliteration of a given Roman script input.

Example:
```sh
curl -X POST "http://localhost:8000/transliterate" -H "Content-Type: application/json" -d '{"text": "namaste"}'
```

## API Endpoints

### POST /transliterate
Transliterates the given Roman script text to Devanagari script.

- **Request Body**:
  ```json
  {
    "text": "namaste"
  }
  ```
- **Response**:
  ```json
  {
    "transliteration": "नमस्ते"
  }
  ```

## Monitoring

### Prometheus and Grafana
Prometheus is used to scrape metrics, and Grafana is used to visualize these metrics.

1. **Access Prometheus**: `http://localhost:9090`
2. **Access Grafana**: `http://localhost:3000`
   - Default credentials: `admin/admin`
   - Configure Prometheus as a data source in Grafana and import the dashboard for visualization.

## Configuration Files

### prometheus.yml
Configuration file for Prometheus to scrape metrics.

### input_vocab.json and target_vocab.json
These files contain the vocabulary for the input Roman script and the target Devanagari script, respectively.

```json
{
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "a": 3,
    "b": 4,
    ...
}
```

```json
{
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "अ": 3,
    "आ": 4,
    ...
}
```

## Acknowledgements
This project uses the following libraries and tools:
- FastAPI
- Prometheus
- Grafana
- Docker
- Torch


