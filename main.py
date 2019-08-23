import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning in TF's prints
    import tensorflow as tf


from squeezenet.train_squeezenet import run

if __name__ == '__main__':
    run()
