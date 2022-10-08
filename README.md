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

Sources: 
- [Creating Lambda container images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [Deploying Lambda functions as container images](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-images.html)
- [Notebook with detailed steps to deploy to AWS Lambda](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022/blob/main/notebooks/lab99_serverless_aws.ipynb)


### Running the deploy process from start to finish

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


- If you’re on an M1 Mac, from `model_server/`, run
```
make full_lambda_deploy_m1
```
- Else, from `model_server/`
```
make full_lambda_deploy
```
