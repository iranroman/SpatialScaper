# Parameters used in the data generation process.
import yaml

def get_params(argv='1'):
    with open('./yaml/gen_params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    # Please define another ./yaml/*.yaml: no more hard coding below please :P
    return params
