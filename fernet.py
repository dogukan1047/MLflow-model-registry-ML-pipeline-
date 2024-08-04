from cryptography.fernet import Fernet

key = Fernet.generate_key()
print(key.decode())  # Save this key and use it in your configuration
