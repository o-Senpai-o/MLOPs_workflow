# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# set working directory as /app 
WORKDIR /app

# copy everything from current directory to app directory in docker container
COPY src/project/prod  /app/prod
COPY requirements.txt /app/prod
ENV PYTHONPATH="/app/prod"

# change the working directory to /app/prod
WORKDIR /app/prod

# Install pip requirements
RUN pip install -r requirements.txt




# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["python", "test_pipeline.py"]