FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements/requirements-backend.txt  ./requirements.txt
RUN  pip3 install -r requirements.txt

# Copy function code
COPY sign_recognizer/ ./sign_recognizer
COPY api_serverless/api.py ./api.py

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api.handler" ]