from IPython.display import clear_output

def update_progress(progress, clear=True, comment=''):
    bar_length = 50
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    if clear:
        clear_output(wait = True)
    text = 'Progress ' + str(comment) + ': [{0}] {1:.1f}%'.format("#" * block + "-" * (bar_length - block),
                                                                  progress * 100)
    print(text)