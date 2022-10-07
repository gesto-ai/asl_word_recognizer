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

build_image:
	docker build -t $(LAMBDA_AND_CONTAINER_NAME) . --file $(AWS_DOCKERFILE_NAME)

build_image_m1:
	docker build -t $(LAMBDA_AND_CONTAINER_NAME) . --file $(AWS_DOCKERFILE_NAME) --platform=linux/amd64

run_container: build_image
	docker run -p 9000:8080 $(LAMBDA_AND_CONTAINER_NAME):latest

run_container_m1: build_image_m1
	docker run -p 9000:8080 $(LAMBDA_AND_CONTAINER_NAME):latest --platform=linux/amd64

authenticate_ecr:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URI)

create_ecr_repository: authenticate_ecr
	aws ecr create-repository --repository-name $(LAMBDA_AND_CONTAINER_NAME) --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE

deploy_to_ecr: build_image authenticate_ecr
	docker tag  $(LAMBDA_AND_CONTAINER_NAME):latest $(IMAGE_URI):latest
	docker push $(IMAGE_URI):latest

deploy_to_ecr_m1: build_image_m1 authenticate_ecr
	docker tag  $(LAMBDA_AND_CONTAINER_NAME):latest $(IMAGE_URI):latest
	docker push $(IMAGE_URI):latest

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

	aws lambda update-function-configuration \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--region $(AWS_REGION) \
	--timeout 60 \
	--memory-size 10240

	aws lambda create-function-url-config \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--auth-type NONE \
	--cors '{"AllowOrigins": ["*"], "AllowCredentials": false}'

	aws lambda add-permission \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--action lambda:invokeFunctionUrl \
	--statement-id "open-access" \
	--principal "*" \
	--function-url-auth-type NONE

get_lambda_url:
	aws lambda get-function-url-config --function-name $(LAMBDA_AND_CONTAINER_NAME) | jq .FunctionUrl

# Test the lambda function we created with a sample payload
test_lambda:
	aws lambda invoke \
	--function-name $(LAMBDA_AND_CONTAINER_NAME) \
	--invocation-type RequestResponse \
	--payload '{"video_url": "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"}' \
	--cli-binary-format raw-in-base64-out lambda.out

	cat lambda.out
