# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

# How to run?

### STEPS:

Clone the repository

```bash
git clonehttps://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & Gemini credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Techstack Used:

- Python
- LangChain
- Flask
- Gemini
- Pinecone

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

    #with specific access

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws


    #Description: About the deployment

    1. Build docker image of the source code

    2. Push your docker image to ECR

    3. Launch Your EC2

    4. Pull Your image from ECR in EC2

    5. Lauch your docker image in EC2

    #Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image

    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:

    #optinal

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

# 6. Configure EC2 as self-hosted runner:

    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO
- PINECONE_API_KEY
- GEMINI_API_KEY

## Production Docker Deployment on AWS EC2

### 1. Build and test locally

```bash
docker build -t medical-chatbot:latest .
docker run --rm -p 8080:8080 \
	-e PINECONE_API_KEY="your_pinecone_key" \
    -e GEMINI_API_KEY="your_gemini_key" \
	medical-chatbot:latest
```

Open http://localhost:8080

### 2. Push image to Amazon ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag medical-chatbot:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot:latest
```

### 3. Run on EC2

SSH to your EC2 host and run:

```bash
docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot:latest
docker stop medical-chatbot || true
docker rm medical-chatbot || true
docker run -d \
	--name medical-chatbot \
	--restart always \
	-p 8080:8080 \
	-e PINECONE_API_KEY="your_pinecone_key" \
    -e GEMINI_API_KEY="your_gemini_key" \
	<account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot:latest
```

### 4. One-time Pinecone indexing

Run this once (locally or on EC2) before chat traffic:

```bash
docker run --rm \
	-e PINECONE_API_KEY="your_pinecone_key" \
    -e GEMINI_API_KEY="your_gemini_key" \
	<account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot:latest \
	python store_index.py
```

### 5. EC2 networking checklist

- Security Group inbound allows TCP 8080 from your source (or via ALB)
- EC2 has outbound internet access to Gemini, Pinecone, and HuggingFace
- If using an ALB, route HTTP 80 -> target group port 8080
