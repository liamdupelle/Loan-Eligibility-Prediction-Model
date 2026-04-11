# import logging module
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# error logging decorator that handles errors
def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        logger.info(f"Started Running Function : {func.__name__}")
    
        try:
            result = func(*args, **kwargs)
        
            logger.info(f"Finished Running Function : {func.__name__}")
            
            return result
        
        # handle value errors, type errors, and index errors    
        except (ValueError, TypeError, IndexError) as e:
            logger.error("%s in %s : %s", type(e).__name__,func.__name__,e)
            return None
        
        # handle file not found errors
        except FileNotFoundError as e:
            logger.error("Input File Not Found in %s : %s", func.__name__,e)
            return None
            
        # handle invalid key errors
        except KeyError as e:
            logger.error("Invalid Key in %s : %s", func.__name__,e)
            return None
        
        # handle permission errors for file
        except PermissionError as e:
            logger.error("Permission to Access File Denied in %s : %s", func.__name__,e)
            return None
        
        # handle unexpected errors
        except Exception as e:
            logger.error("%s in %s : %s", type(e).__name__, func.__name__, e)
            return None
    return wrapper