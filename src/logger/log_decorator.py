# import logging module
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# error logging decorator that handles errors
def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        logger.info(f"Started Running Function : {func.__name__}",)
    
        try:
            result = func(*args, **kwargs)
        
            logger.info(f"Finished Running Function : {func.__name__}",)
            
            return result
            
        except (ValueError, TypeError, IndexError) as e:
            logger.error("%s in %s : %s", type(e).__name__,func.__name__,e)
            return None
        
        except FileNotFoundError:
            logger.error("Input File Not Found in Function : %s",func.__name__)
            return None
            
        except KeyError:
            logger.error("Invalid Key in Function : %s",func.__name__)
            return None
        
        except PermissionError:
            logger.error("Permission to Access File Denied in Function : %s",func.__name__)
            return None
        
        except:
            logger.error("Unexpected Error Occurred in Function : %s",func.__name__)
    return wrapper