
import os
os.environ['TORCH_HOME'] = '/projects/wang/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/projects/wang/.cache'

my_variable = os.environ.get('TORCH_HOME')