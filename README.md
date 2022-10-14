# Model Serve

## Pre-requisites

- Python 3.9
- A virtual environment

In a virtual environment (this was tested in a `conda` environment), install the required packages from the `requirements.txt` file:
```
pip install -r requirements/requirements-frontend.txt
```

## Using the Sign Recognizer script

### Step 1: Get the model weights from the Lambda Labs VM

The script currently expects the model weights to be stored in `model_serve/artifacts/sign-recognizer`.

```
cd model_serve

scp -r team_046@150.136.219.226:/home/team_046/models/artifacts .  
```


### [NOT WORKING] Step 2: run `word_sign_recognizer.py`

```
python sign_recognizer/word_sign_recognizer.py /path/to/video.mp4
```

You can download a sample video file from [here](https://discord.com/channels/@me/1017414703133237298/1024513846427262976).


## Running the Streamlit app (connected to the Sign Recognizer)

After you have gotten the model weights from Step 1 of running the sign recognizer script, run:

```
streamlit run app.py
```

## Building and testing the backend prediction server
We can test the prediction server logic without deploying to AWS.

0. First, comment out the line `include .env` in the Makefile, unless you have a `.env` file with the expected information.
1. Build the docker image
```
# If you're on an M1 mac, run `make build_m1` instead
make build
```
2. Run the container locally
```
make run
```
3. In a different terminal session, run a local test with the demo video by sending a POST request to the running container.

**NOTE:** If you run this in your local machine, it will probably timeout - so make sure to run it in a GPU-powered machine!
```
make test_local
```

## Deploy model code to AWS ECR/Lambda

Sources: 
- [Creating Lambda container images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [Deploying Lambda functions as container images](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-images.html)
- [Notebook with detailed steps to deploy to AWS Lambda](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022/blob/main/notebooks/lab99_serverless_aws.ipynb)

### Running the deploy process from start to finish (Currently not supported unless you have an AWS acccount connected)

Pre-pre requisites:
- Have an AWS account, with a secret key/secret access key

Pre-requisites (for the `sign-recognizer` app, these have already been created):
- Create a repository in the AWS Elastic Container Registry (ECR)
```
make create_ecr_repository
```
- Create an AWS Lambda function (will take a bit and may ran into timeouts)
```
make create_lambda
```

The following commands will:
- Build a Docker image
- Push it to AWS ECR
- Update the Lambda function with the newest image
- Get the status of the Lambda function update


- If you’re on an M1 Mac, run:
```
make full_lambda_deploy_m1
```
- Else, run:
```
make full_lambda_deploy
```
