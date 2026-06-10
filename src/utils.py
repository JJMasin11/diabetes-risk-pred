import os
import sys
import joblib
from io import BytesIO
from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def get_vault_secret(secret_name: str) -> str:
    # Load environment variables
    load_dotenv()

    # Construct vault URI
    vault_uri = f"https://{os.getenv('VAULT_NAME')}.vault.azure.net/"

    credential = DefaultAzureCredential()

    # Initialize SecretClient
    client = SecretClient(vault_url=vault_uri, credential=credential)

    try:
        # Fetch secret
        secret = client.get_secret(secret_name)

        return secret.value
    
    except Exception as e:
        raise CustomException(e, sys)
    
def upload_to_blob(file_path, name: str):
    # Initialize connection client
    connection_string = get_vault_secret('STORAGE-CONN')
    service_client = BlobServiceClient.from_connection_string(connection_string)

    # Define container
    container_name = os.getenv('CONTAINER_NAME')

    # Get blob client
    client = service_client.get_blob_client(container=container_name, blob=name)
    
    # Upload
    try:
        logging.info(f"Uploading {name} to Azure Blob Storage...")
        with open(file_path, "rb") as data:
            client.upload_blob(data, overwrite=True)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_from_blob(name: str):
    # Initialize connection client
    connection_string = get_vault_secret('STORAGE-CONN')
    service_client = BlobServiceClient.from_connection_string(connection_string)

    # Define container
    container_name = os.getenv('CONTAINER_NAME')

    # Get blob client
    client = service_client.get_blob_client(container=container_name, blob=name)

    # Download blob
    obj = client.download_blob().readall()

    # Load model
    return joblib.load(BytesIO(obj))