# Model Serve

## Pre-requisites

In a virtual environment (this was tested in a `conda` environment), install the required packages from the `requirements.txt` file:
```
pip install -r requirements.txt
```

## Steps to use the Sign Recognizer script

### Step 1: Get the model weights from the Lambda Labs VM

The script currently expects the model weights to be stored in `model_serve/models/`.

```
cd model_serve

scp -r team_046@150.136.219.226:/home/team_046/models ./
```


### Step 2: run `sign_recognizer.py`

```
python sign_recognizer.py /path/to/video.mp4
```

You can download a sample video file from [here](https://discord.com/channels/@me/1017414703133237298/1024513846427262976).


## Steps to run the Streamlit app (connected to the Sign Recognizer)

After you have gotten the model weights from Step 1 of running the sign recognizer script, run:

```
streamlit run app.py
```

## Steps to deploy model code

Source: [https://docs.aws.amazon.com/lambda/latest/dg/images-create.html](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)

### 1. Make sure you have configured your AWS credentials in your terminal and run the login process
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [YOUR_USER_ID].dkr.ecr.us-east-1.amazonaws.com    
```

### 2. Building the Docker image
- If you’re on an M1 Mac, from `model_server/`, run
```
docker build -t sign_recognizer . --file api_serverless/Dockerfile --platform=linux/amd64
```
- Else, from `model_server/`, run
```
docker build -t sign_recognizer . --file api_serverless/Dockerfile
```

### 2.1 Test your image before pushing it to Amazon's Elastic Container Registry (ECR)
- On one terminal, run `docker run -p 9000:8080 sign_recognizer:latest`
 (add the `--platform=linux/amd64` if you're on an M1 Mac)
- On another terminal, run `curl -i -XPOST \
  "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"video_url": "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"}'`. Make sure it runs!

### 3. Tag your image!
```
docker tag sign_recognizer:latest [YOUR_USER_ID].dkr.ecr.us-east-1.amazonaws.com/sign_recognizer:latest
```

### 4. Push your image
```
docker push [YOUR_USER_ID].dkr.ecr.us-east-1.amazonaws.com/sign_recognizer:latest
```

### 5. Create a Lambda function

- AWS Docs https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-images.html
- More instructions: [https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022/blob/main/notebooks/lab99_serverless_aws.ipynb](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022/blob/main/notebooks/lab99_serverless_aws.ipynb)