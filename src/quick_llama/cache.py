import hashlib
import pickle
import os

# Attempt to determine a writable temporary directory.
try:
	script_dir = os.path.abspath(os.path.dirname(__file__))
	TMP_DIR = os.path.join(script_dir, "tmp")
	os.makedirs(TMP_DIR, exist_ok=True)
	test_path = os.path.join(TMP_DIR, "test.txt")
	# with open(test_path, "w") as f:
	# 	f.write("test")
	# os.remove(test_path)
except Exception as e:
	TMP_DIR = None
	for _ in range(5):
		print(f"*" * 60,)
		print(f"  cache.py  WARNING: Could not create a temporary directory for storing results.", flush=True)
		print(f"*" * 60,)
	raise Exception(f"  cache.py  WARNING: Could not create a temporary directory for storing results.")

def obj_to_int_32(obj):
	"""
	Convert any object to a 32-bit integer hash via SHA-256 of its string form.
	Useful for reproducible seeds from arbitrary Python objects.

	Args:
		obj (Any): The object to be hashed.

	Returns:
		int: The lower 32 bits of the SHA-256 hash of the object's string representation.
	"""
	obj_str = str(obj)
	hash_str = hashlib.sha256(obj_str.encode('utf-8')).hexdigest()
	# Take lower 32 bits
	seed_int = int(hash_str, 16) & 0xFFFFFFFF
	return seed_int


def obj_to_str_hash(obj):
	"""
	Convert any object to a full SHA-256 hash string of its string form.

	Args:
		obj (Any): The object to be hashed.

	Returns:
		str: Hex digest string representing SHA-256 hash.
	"""
	obj_str = str(obj)
	hash_str = hashlib.sha256(obj_str.encode('utf-8')).hexdigest()
	return hash_str

def quick_key(obj):
	"""
	Generate a cache key for the given object.
	"""
	return obj_to_str_hash(obj)

def quick_save(key=None, obj=None):
	"""
	Serialize (pickle) and save 'obj' into TMP_DIR under the filename '{key}.cache'.

	Args:
		obj (Any): The Python object to be pickled.
		key (str): The base name of the cache file.

	Returns:
		None
	"""
	if TMP_DIR is None:
		return
	path = os.path.join(TMP_DIR, f"{key}.cache")
	os.makedirs(TMP_DIR, exist_ok=True)
	with open(path, 'wb') as f:
		pickle.dump(obj, f)


def quick_load(name):
	"""
	Load and return the object stored in TMP_DIR under '{name}.cache'.

	Args:
		name (str): The base name (without extension) of the cache file.

	Returns:
		Any: The Python object that was previously saved.

	Raises:
		FileNotFoundError: If the cache file does not exist.
	"""
	if TMP_DIR is None:
		raise Exception("We don't have a temporary directory!")
	
	path = os.path.join(TMP_DIR, f"{name}.cache")
	if not os.path.exists(path):
		raise FileNotFoundError(f"{path} does not exist!")
	with open(path, 'rb') as f:
		return pickle.load(f)
