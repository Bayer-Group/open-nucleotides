import logging

from nucleotides import config
from nucleotides.model.pinkpanther import PinkPanther
from nucleotides.model.woodpecker import Woodpecker

model_classes = {
    "pinkpanther": PinkPanther,
    "woodpecker": Woodpecker,
}


def get_model():
    if config.INFERENCE_MODEL is None:
        raise ValueError('''INFERENCE_MODEL can not be None
                         Set the INFERENCE_MODEL setting by one of the following methods:
                         
                         1) Set an environment value manually before running a command:
    
                            NUCLEOTIDES_INFERENCE_MODEL=/path/to/checkpoint [COMMAND]
                        
                        2) Add a line to a .env file:
                        
                            NUCLEOTIDES_INFERENCE_MODEL=/path/to/checkpoint
                            
                        3) Change the value in nucleotides/settings.py
                        
                        ''')
    if config.INFERENCE_MODEL_CLASS not in model_classes.keys():
        raise ValueError('''INFERENCE model can not be {}, but has to be one of: {}
                         Set the INFERENCE_MODEL_CLASS setting by one of the following methods:

                         1) Set an environment value manually before running a command:

                            NUCLEOTIDES_INFERENCE_MODEL_CLASS="model_class" [COMMAND]

                        2) Add a line to a .env file:

                            NUCLEOTIDES_INFERENCE_MODEL_CLASS="model_class"

                        3) Change the value in nucleotides/settings.py

                        '''.format(config.INFERENCE_MODEL, model_classes.keys()))

    logging.info(
        f"Loading model of class {config.INFERENCE_MODEL_CLASS} from checkpoint file {config.INFERENCE_MODEL}"
    )
    return model_classes[config.INFERENCE_MODEL_CLASS].load_from_checkpoint(
        config.INFERENCE_MODEL
    )
