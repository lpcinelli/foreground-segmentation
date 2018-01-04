'''
config_utils.py

Configuration file treatment files.
'''

import yaml

class Bunch(object):
    '''
    Converts dict to namespace
    '''
    def __init__(self, adict):
        '''
        Initializes a namespace given a dictionary.

        @param adict target dictionary.

        @return converted object.
        '''
        self.__dict__.update(adict)

def create_config_file(path, args=None):
    '''
    Creates a configuration file.

    @param path file path.
    @param args configuration arguments.
    '''

    if args:
        # Use provided arguments.
        args = vars(args)
    else:
        # Set parameters dictionary with arguments
        args = {}
        args['img_path'] = '' # path to dataset
        args['augmentation'] = '' #TODO use data augmentation
        args['train'] = '' # path to train.csv
        args['val'] = '' # path to val.csv
        args['arch'] = 'toynet' # model architecture
        args['arch_params'] = [] # model architecture params
        args['loss'] = 'bce' # losses
        args['workers'] = 4 # number of data loading workers
        args['epochs'] = 90 # number of total epochs to run
        args['start_epoch'] = 0 # manual epoch number (useful on restarts)
        args['batch_size'] = 32 # manual epoch number (useful on restarts)
        args['optim'] = 'adam' # algorithm for model optimization
        args['learning_rate'] = 0.1 # initial learning rate
        args['momentum'] = 0.9 # momentum
        args['weight_decay'] = 1e-4 # weight decay
        args['print_freq'] = 10 # print frequency
        args['resume'] = '' # path to latest checkpoint
        args['evaluate'] = True # evaluate model on validation set
        args['cuda'] = True # use GPU
        args['visdom'] = True # use visdom
        args['save_folder'] = 'models/' # Location to save models

    # Saving on yaml
    with open(path, 'w') as yaml_file:
        yaml.dump(args, stream=yaml_file, default_flow_style=False)

def load_config_file(path):
    '''
    Loads configuration file.

    @param file path.

    @return models arguments.
    '''

    # Open yaml file
    with open(path, 'r') as yaml_file:
        args = yaml.load(yaml_file)

        # Converting to namespace and returning
        return Bunch(args)
    return None
