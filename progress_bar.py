import logging
import time

logger = logging.getLogger(__name__)

def progress_bar(total_steps, message=''):
    """This generator gives regular progress reports. Usage:
    pbar = progressbar(len(steps)), "Doing steps...")
    for step in steps:
        ...
        pbar.next()
    """
    logger.info('%s...', message)
    step = 0
    start = time.clock()
    previous_notif = start
    virgin = True
    while step < total_steps - 1:
        now = time.clock()
        elapsed = now - start
        if (now - previous_notif) > 10:
            logger.info("%s/%s complete after %d seconds elapsed", 
                        step, total_steps, round(elapsed))
            previous_notif = now
        elif (virgin is True and elapsed > 1):
            virgin = False
            logger.info("%s/%s complete after %d second elapsed", 
                        step, total_steps, round(elapsed))
        yield
        step += 1
    logger.info("Completed in %d seconds.", round(time.clock() - start))
    yield
