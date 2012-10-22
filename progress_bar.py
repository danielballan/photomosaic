import logging
import time

logger = logging.getLogger(__name__)

def progress_bar(total_steps, message=''):
    logger.info('%s...', message)
    step = 0
    start = time.clock()
    previous_notif = start
    while step < total_steps - 1:
        if (time.clock() - previous_notif) > 10:
             elapsed = time.clock() - start
             logger.info("%s/%s complete after %d seconds elapsed", 
                         step, total_steps, int(elapsed))
             previous_notif = time.clock()
        yield
        step += 1
    logger.info("Complete.")
    yield
