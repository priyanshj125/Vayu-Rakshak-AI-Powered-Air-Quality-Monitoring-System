import os
import json
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# The connection string provided by the user
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

HOT_CONTAINER_NAME = "vayu-hot-storage"
COLD_CONTAINER_NAME = "vayu-cold-storage"

blob_service_client = None
hot_container_client = None
cold_container_client = None

def get_blob_service_client():
    global blob_service_client
    if blob_service_client is None:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    return blob_service_client

def initialize_containers():
    """Ensure that the hot and cold storage containers exist."""
    global hot_container_client, cold_container_client
    try:
        client = get_blob_service_client()
        
        # Hot storage container
        hot_container_client = client.get_container_client(HOT_CONTAINER_NAME)
        if not hot_container_client.exists():
            hot_container_client = client.create_container(HOT_CONTAINER_NAME)
            logger.info(f"Created Azure container: {HOT_CONTAINER_NAME}")
        
        # Cold storage container
        cold_container_client = client.get_container_client(COLD_CONTAINER_NAME)
        if not cold_container_client.exists():
            cold_container_client = client.create_container(COLD_CONTAINER_NAME)
            logger.info(f"Created Azure container: {COLD_CONTAINER_NAME}")
    except Exception as e:
        logger.error(f"Error initializing Azure containers: {e}")

def upload_reading_hot(payload: dict):
    """Upload a reading to hot storage."""
    try:
        client = get_blob_service_client()
        hot_client = client.get_container_client(HOT_CONTAINER_NAME)
        
        sensor_id = payload.get("sensor_id", "unknown")
        timestamp = payload.get("timestamp", datetime.utcnow().isoformat())
        # Clean timestamp for filename: replace spaces and colons
        ts_clean = str(timestamp).replace(":", "-").replace(" ", "_")
        blob_name = f"{sensor_id}/{ts_clean}.json"
        
        blob_client = hot_client.get_blob_client(blob_name)
        
        data_str = json.dumps(payload)
        blob_client.upload_blob(data_str, overwrite=True)
        logger.info(f"Uploaded reading to hot storage blob: {blob_name}")
    except Exception as e:
        logger.error(f"Error uploading reading to hot storage: {e}")

def archive_to_cold(limit: int = 100):
    """Move blobs from Hot container to Cold container for long-term archival."""
    archived_count = 0
    try:
        client = get_blob_service_client()
        hot_client = client.get_container_client(HOT_CONTAINER_NAME)
        cold_client = client.get_container_client(COLD_CONTAINER_NAME)
        
        # List blobs in hot storage
        blobs = hot_client.list_blobs()
        
        for idx, blob in enumerate(blobs):
            if idx >= limit:
                break
            
            blob_name = blob.name
            
            # Copy to cold storage
            source_blob = hot_client.get_blob_client(blob_name)
            dest_blob = cold_client.get_blob_client(blob_name)
            
            dest_blob.start_copy_from_url(source_blob.url)
            
            # Delete from hot storage
            source_blob.delete_blob()
            
            archived_count += 1
            logger.info(f"Archived blob: {blob_name} to {COLD_CONTAINER_NAME}")
            
    except Exception as e:
        logger.error(f"Error archiving to cold storage: {e}")
    
    return archived_count
