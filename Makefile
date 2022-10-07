# NOTE: Asume .env contains
# AWS_ACCOUNT_ID=123456789
# AWS_REGION=some-valid-aws-region
# AWS_ACCESS_KEY=AKBCDEFGHIJKL
# AWS_SECRET_ACCESS_KEY=qwertyuiopasdfghjklzxcvbnm

include .env

LAMBDA_AND_CONTAINER_NAME = sign_recognizer
LAMBDA_ROLE_NAME = sign_recognizer-role

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
