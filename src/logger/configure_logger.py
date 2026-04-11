# import logging module
import logging

def configure_logger():
    
    # configure logger
    logging.basicConfig(filename = 'log.txt',
                    filemode = 'w',
                    level = logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s",
                   force = True)
