'''
Use this to more easily use Accelerate

accelerate launch --config_file accelerate_config.yaml train_wrapper.py

'''

if __name__ == "__main__":
	from src import train
	train.main()