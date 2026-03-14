from azure_storage import initialize_containers, get_blob_service_client
from dotenv import load_dotenv

load_dotenv()

def test_azure():
    print("Initializing containers...")
    initialize_containers()
    
    client = get_blob_service_client()
    containers = list(client.list_containers())
    print("\nContainers available:")
    for c in containers:
        print(f" - {c.name}")

if __name__ == "__main__":
    test_azure()
