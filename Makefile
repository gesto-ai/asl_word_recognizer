#### Makefile template taken from: https://github.com/caseyfitz/cookiecutter-disco-pie/blob/main/%7B%7Bcookiecutter.repo_name%7D%7D/Makefile

# NOTE: Asume .env contains
# AWS_ACCOUNT_ID=123456789
# AWS_REGION=some-valid-aws-region
# AWS_ACCESS_KEY=AKBCDEFGHIJKL
# AWS_SECRET_ACCESS_KEY=qwertyuiopasdfghjklzxcvbnm

include .env

LAMBDA_AND_CONTAINER_NAME = sign-recognizer
LAMBDA_ROLE_NAME = sign-recognizer-role

ECR_URI = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
IMAGE_URI = $(ECR_URI)/$(LAMBDA_AND_CONTAINER_NAME)
AWS_DOCKERFILE_NAME = api_serverless/Dockerfile

###########################
# General docker commands
###########################

build:
	docker build -t $(LAMBDA_AND_CONTAINER_NAME) . --file $(AWS_DOCKERFILE_NAME)

build_m1:
	docker build -t $(LAMBDA_AND_CONTAINER_NAME) . --file $(AWS_DOCKERFILE_NAME) --platform=linux/amd64

run:
	docker run -p 9000:8080 $(LAMBDA_AND_CONTAINER_NAME):latest

test_local:
	curl -i -XPOST \
	"http://localhost:9000/2015-03-31/functions/function/invocations" \
	-d '{"video_url": "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"}'

######################
# AWS ECR commands
######################
authenticate_ecr:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URI)

create_ecr_repository: authenticate_ecr
	aws ecr create-repository --repository-name $(LAMBDA_AND_CONTAINER_NAME) --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE

deploy_to_ecr: build authenticate_ecr
	docker tag  $(LAMBDA_AND_CONTAINER_NAME):latest $(IMAGE_URI):latest
	docker push $(IMAGE_URI):latest

deploy_to_ecr_m1: build_m1 authenticate_ecr
	docker tag  $(LAMBDA_AND_CONTAINER_NAME):latest $(IMAGE_URI):latest
	docker push $(IMAGE_URI):latest

# Fully build an image and deploy it to AWS ECR
full_ecr_deploy: build deploy_to_ecr
full_ecr_deploy_m1: build_m1 deploy_to_ecr_m1

######################
# AWS Lambda commands
######################
create_lambda_role:
	aws iam create-role \
	--role-name $(LAMBDA_ROLE_NAME) \
	--assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
	
	aws iam attach-role-policy \
	--role-name $(LAMBDA_ROLE_NAME) \
	--policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

	aws iam attach-role-policy \
	--role-name $(LAMBDA_ROLE_NAME) \
	--policy-arn arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess

# Create lambda function, increase timeout, max out memory, and create public function URL
create_lambda_function:
	echo "wait 10 seconds for role..."	
		$(shell sleep 10)
		aws lambda create-function \
		--function-name $(LAMBDA_AND_CONTAINER_NAME) \
		--region $(AWS_REGION) \
		--package-type Image \
		--code ImageUri=$(IMAGE_URI):latest \
		--role $(shell aws iam get-role --role-name $(LAMBDA_ROLE_NAME) --output json | jq -r '.Role.Arn')
	
	echo "wait 5 seconds for function to be created..."	
	$(shell sleep 5)
	aws lambda update-function-configuration \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--region $(AWS_REGION) \
	--timeout 60 \
	--memory-size 10240

	echo "wait 5 seconds for configuration to be updated..."	
	$(shell sleep 5)
	aws lambda create-function-url-config \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--auth-type NONE \
	--cors '{"AllowOrigins": ["*"], "AllowCredentials": false}'

	echo "wait 5 seconds for creating the url..."	
	$(shell sleep 5)
	aws lambda add-permission \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--action lambda:invokeFunctionUrl \
	--statement-id "open-access" \
	--principal "*" \
	--function-url-auth-type NONE

get_lambda_url:
	aws lambda get-function-url-config \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) | jq .FunctionUrl

get_lambda_state:
	aws lambda get-function \
	--function-name sign-recognizer | jq .Configuration.State

get_lambda_update_status:
	aws lambda get-function \
	--function-name sign-recognizer | jq .Configuration.LastUpdateStatus

update_lambda:
	aws lambda update-function-code --function-name $(LAMBDA_AND_CONTAINER_NAME) --image-uri $(IMAGE_URI):latest

# Test the lambda function we created with a sample payload
test_lambda:
	echo "If you're running this for the first time or after an update, it will take a bit to initialize the function!"
	aws lambda invoke \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--invocation-type RequestResponse \
	--payload '{"video_url": "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"}' \
	--cli-binary-format raw-in-base64-out lambda.out

	cat lambda.out

# Run the full deploy pipeline: build image, push it to ECR, update the lambda function code, and confirm that the function is being updated
full_lambda_deploy: full_ecr_deploy update_lambda get_lambda_update_status
full_lambda_deploy_m1: full_ecr_deploy_m1 update_lambda get_lambda_update_status
