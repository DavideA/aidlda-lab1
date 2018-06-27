import time

term_width = 211

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(stage, current, total, loss_str=None):
    global last_time, begin_time
    if current == 0:
        print('')
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    bar = ''
    bar += stage
    bar += ' ['
    for i in range(cur_len):
        bar += '='
    bar += '>'
    for i in range(rest_len):
        bar += '.'
    bar += ']'

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    msg = ''
    msg += ' Step: %s' % format_time(step_time)
    msg += ' | Tot: %s' % format_time(tot_time)
    if bar:
        msg += ' | ' + bar

    msg = msg + ' ' + loss_str
    print('\r' + msg, end='', flush=True)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
