# AWS Health AI Platform

An end-to-end **MLOps and Agentic AI healthcare platform** built with Amazon SageMaker, Amazon Bedrock, Bedrock AgentCore, OpenSearch Serverless, FastAPI, Docker, and CI/CD.

The platform combines a machine learning heart disease risk model with Retrieval-Augmented Generation (RAG) over cardiovascular clinical guidelines and an AI agent capable of orchestrating both capabilities.

---

## Architecture

### ML Pipeline

```text
Heart Disease Dataset
 ↓
Preprocessing
 ↓
SageMaker Training
 ↓
Model Evaluation
 ↓
Quality Gate
 ↓
SageMaker Model Registry
 ↓
Registered Model
```

### RAG Ingestion

```text
Clinical Guideline PDF
 ↓
Amazon S3
 ↓
Amazon Textract
 ↓
Extracted Text
 ↓
Chunking
 ↓
Amazon Titan Embeddings
 ↓
OpenSearch Serverless
 ↓
Vector Index
```

### RAG

```text
User Question
 ↓
FastAPI /ask
 ↓
Guardrail
 ↓
Titan Embedding
 ↓
OpenSearch Vector Search
 ↓
Top 3 Chunks
 ↓
Amazon Bedrock
 ↓
Grounded Answer + Sources
```

### Agent

```text
User Request
 ↓
FastAPI /agent
 ↓
Amazon Bedrock AgentCore
 ↓
Agent Orchestration
 ↓
┌──────────────────────┬──────────────────────┐
│                      │                      │
ML Risk Prediction     Guideline Search
│                      │
└──────────┬───────────┘
           ↓
     Agent Response
```

---

## Clinical Guideline Source

The RAG knowledge base uses the **2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease – Guidelines Made Simple**, published by the American College of Cardiology.

The source document is not stored in this repository.

[View the official ACC guideline](https://www.acc.org/-/media/Non-Clinical/Files-PDFs-Excel-MS-Word-etc/Guidelines/2019/B19193-Prevention-GMS-Tool.pdf)

To reproduce the RAG knowledge base:

1. Download the guideline from the official ACC source.
2. Upload the PDF to the project Amazon S3 bucket.
3. Run the ingestion pipeline:

```powershell
python rag/extract_textract.py
python rag/chunk_documents.py
python rag/embed_chunks.py
python rag/index_chunks.py
```

The ingestion flow is:

```text
PDF in S3
 ↓
Amazon Textract
 ↓
Extracted Document
 ↓
38 Chunks
 ↓
Titan Embeddings
 ↓
OpenSearch Serverless
```

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Web Server | Uvicorn |
| ML | Scikit-learn |
| ML Platform | Amazon SageMaker |
| LLM | Amazon Bedrock |
| Embeddings | Amazon Titan Embeddings |
| Agent | Amazon Bedrock AgentCore |
| Vector Database | Amazon OpenSearch Serverless |
| Document Extraction | Amazon Textract |
| Guardrails | Amazon Bedrock Guardrails |
| Serverless Tools | AWS Lambda |
| Object Storage | Amazon S3 |
| Containers | Docker |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Authentication | GitHub OIDC |
| Image Registry | Amazon ECR |
| Cloud | Amazon ECS / AWS Fargate |
| Infrastructure | Terraform |

---

## Project Structure

```text
aws-health-ai-platform/
│
├── app/
│   ├── api.py
│   └── guardrails.py
│
├── agent/
│   └── search_guidelines/
│
├── data/
│
├── deployment/
│   ├── deploy.py
│   └── inference.py
│
├── evaluation/
│   ├── evaluate.py
│   └── evaluation.json
│
├── infrastructure/
│
├── pipelines/
│
├── rag/
│   ├── extract_textract.py
│   ├── chunk_documents.py
│   ├── embed_chunks.py
│   ├── index_chunks.py
│   ├── search.py
│   └── llm.py
│
├── tests/
│
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Run the Project

The project can run in three ways:

1. Local Python
2. Docker
3. AWS Cloud

---

## 1. Local Python

### Requirements

- Python 3.12+
- AWS CLI
- AWS credentials
- Required AWS services
- Required environment variables
- RAG knowledge base indexed in OpenSearch Serverless

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Configure AWS credentials:

```powershell
aws configure --profile cristhian-dev
```

Configure the required environment variables for Bedrock, AgentCore, Guardrails, and OpenSearch.

If the RAG knowledge base has not been created yet, download the clinical guideline, upload it to the project S3 bucket, and run:

```powershell
python rag/extract_textract.py
python rag/chunk_documents.py
python rag/embed_chunks.py
python rag/index_chunks.py
```

Start the API:

```powershell
uvicorn app.api:app --reload
```

Access the application:

```text
API:     http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
Health:  http://127.0.0.1:8000/health
```

---

## 2. Docker

The Docker image packages the FastAPI application while the required AWS services remain external AWS-managed resources.

### Requirements

- Docker
- AWS account
- AWS credentials
- Required AWS services
- Required environment variables
- RAG knowledge base indexed in OpenSearch Serverless

Build the Docker image:

```powershell
docker build -t aws-health-ai-api .
```

Configure the required AWS credentials and environment variables.

Run the container:

```powershell
docker run -p 8000:8000 aws-health-ai-api
```

Access the application:

```text
API:     http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
Health:  http://127.0.0.1:8000/health
```

---

## 3. AWS Deployment

The cloud deployment is automated through GitHub Actions.

Commit and push to `master`:

```powershell
git add .
git commit -m "Update application"
git push origin master
```

The deployment pipeline runs:

```text
Push to master
 ↓
CI
 ↓
Pytest
 ↓
AI Evaluation
 ↓
CD
 ↓
Build Docker Image
 ↓
Tag Image with Git SHA
 ↓
Push to Amazon ECR
 ↓
Deploy to Amazon ECS Express Mode
```

Access the deployed application:

```text
API:     https://<ecs-api-url>
Swagger: https://<ecs-api-url>/docs
Health:  https://<ecs-api-url>/health
```

External users accessing the deployed API do not need an AWS account, AWS credentials, Python, Docker, or the AWS CLI.

---

## API Endpoints

```text
GET  /health
POST /ask
POST /agent
```

`/ask` uses the RAG pipeline to retrieve relevant cardiovascular guideline context and generate a grounded answer with sources.

`/agent` uses the agentic workflow to orchestrate ML heart disease risk prediction and clinical guideline retrieval.

---

## Evaluation

The project includes automated AI evaluation:

```text
Evaluation Dataset
 ↓
API / Agent Execution
 ↓
Response Validation
 ↓
Evaluation Result
 ↓
Pass / Fail
```

Evaluation runs as part of CI before cloud deployment.

---

## CI

The CI pipeline:

```text
Push / Pull Request
 ↓
GitHub Actions Runner
 ↓
Install Dependencies
 ↓
Validate Python
 ↓
Run Pytest
 ↓
Authenticate to AWS with OIDC
 ↓
Run AI Evaluation
 ↓
Pass / Fail
```

GitHub Actions authenticates to AWS through **OpenID Connect (OIDC)** instead of using long-lived AWS access keys.

---

## CD

CD runs only after CI completes successfully.

```text
CI Success
 ↓
GitHub Actions
 ↓
Authenticate to AWS with OIDC
 ↓
Build Docker Image
 ↓
Tag with Git SHA
 ↓
Login to Amazon ECR
 ↓
Push Image
 ↓
Update ECS Express Service
 ↓
New Deployment
```

Separate IAM roles are used for CI and CD following the principle of least privilege.

Docker images are tagged with the Git commit SHA for deployment traceability.

---

## AWS Architecture

```text
AWS
│
├── Amazon S3
│   └── Clinical Guideline PDF
│
├── Amazon Textract
│   └── Document Extraction
│
├── Amazon SageMaker
│   ├── ML Pipeline
│   └── Model Registry
│
├── Amazon Bedrock
│   ├── Foundation Models
│   ├── Titan Embeddings
│   └── Guardrails
│
├── Amazon Bedrock AgentCore
│   └── Agentic Orchestration
│
├── Amazon OpenSearch Serverless
│   └── Clinical Guideline Vector Index
│
├── AWS Lambda
│   └── Agent Tools
│
├── Amazon ECR
│   └── API Docker Images
│
└── Amazon ECS / AWS Fargate
    └── aws-health-ai-api
```

---

## Deployment Summary

```text
LOCAL PYTHON
Python + AWS Services
        ↓
http://127.0.0.1:8000


DOCKER
FastAPI Container + AWS Services
        ↓
http://127.0.0.1:8000


AWS CLOUD
Git Push
   ↓
CI
   ↓
AI Evaluation
   ↓
CD
   ↓
Amazon ECR
   ↓
Amazon ECS Express Mode
   ↓
https://<ecs-api-url>
```

---

## Disclaimer

This project is intended for educational and portfolio purposes. Machine learning predictions and generated responses are not medical diagnoses and should not replace professional clinical judgment.

---

## Author

**Cristhian Balta**