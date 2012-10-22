import logging

logger = logging.getLogger(__name__)

def progress_bar(total_steps, message='', notifications=8):
    step = 0
    logger.info('%s...', message)
    notifications = min(total_steps, notifications)
    while step < total_steps - 1:
        if step % (total_steps // notifications) == 0:
            logger.info('%s/%s', step, total_steps)
        yield
        step += 1
    logger.info('Complete.')
    yield
