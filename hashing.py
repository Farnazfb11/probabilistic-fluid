import hashlib

# Function to hash a string to a numeric ID
#vertex Id gen
def hash_id(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
