# Genre Music Transformer

Simple overview of use/purpose.

## Prerequisites

The developed software for this project runs on Python 3.12.0, which must be installed before you can start running anything. To check if you have Python 3.12.0, run the following command:

```
    python --version
```

If the correct version is shown, then move on to 'Getting Started'. Otherwise, you likely do not have Python or the correct version installed, in which case continue on to the next step.

We recommend using the pyenv version manager to manage Python versions. The guide on pyenv installation for your specific system is provided in the [pyenv GitHub web page](https://github.com/pyenv/pyenv). After pyenv is installed, you can install and use the correct version with the following commands:

```
    pyenv install 3.12.0
    pyenv local 3.12.0
```

We should then be using the correct version of Python.

## Getting Started

Here we detail the steps required to be able to run the program for the first time.

The first step is to initialise and run a virtual environment to install our packages in. The guide to creating a virtual environment is provided in its [Python 3.12.2 documentation](https://docs.python.org/3/library/venv.html). For example, the following commands are for creating and running a virtual environment on a bash shell:

```
    python -m venv venv
    source venv/bin/activate
```

Once the CLI is running a virtual environment, the next step is to install the required packages. To do this run the following command:

```
    pip install -r requirements.txt
```

If you are running a Windows system, you will need to also run the following command to utilise your GPU. This command can be skipped if you are fine with running everything on the CPU.

```
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Once all the packages are installed, you are ready to run the software. Note that the software can only be run when the CLI is running a virtual environment.

Before training, it may be helpful to get acquainted with the configuration (config). This is the file the user modifies to make changes to how the programs function.

```
    ./common/config.py
```

The effects of changing each of the variables in the file are explained in the config file. There are many modifications the user can make to suit their needs, however, the following steps for training or generating will only go over the minimum configuration changes necessary to get the programs running.

## Training

Before you train your MuTr model, you must have a folder of MIDI files in the resources folder to train the model on. The name of the directory is considered the music genre of the files. An example dataset is provided for you.

```
    ./resources/{GENRE}
```

Then modify the config variables to account for the genre of music and the number of files the model will be trained on.

```
    GENRE = {GENRE: str}
    SAMPLE_SIZE = {NUMBER OF FILES: int}
```

Then to train the model, you run the following command:

```
    python train.py
```

This will create a dataset cache and then start the training process. Once the training process is finished, a training and validation loss graph will be shown and a trained MuTr model for each epoch will have been saved in the model sub-directory.

```
    ./bin/model/{GENRE}_{EPOCH}
```

If you want to stop the training before all epochs have finished, simply use a keyboard interrupt. The graph will be shown and the models of completed epochs will have been saved.

## Generating

Before you generate an output MIDI file from a trained model, you must first of course train a model. Refer to the Training step for this, and note the genre and sample size the model was trained on and the epoch of the trained model you want to generate from.

Modify the config variables of the genre, sample size and the picked model epoch, along with a file name for the output MIDI file.

```
    MIDI_FILE_NAME = {FILE NAME: str}
    MODEL_GENRE = {GENRE: str}
    MODEL_SAMPLE_SIZE = {NUMBER OF FILES: int}
    MODEL_EPOCH = {PICKED EPOCH: int}
```

To generate a MIDI file, you run the following command:

```
    python generate.py
```

Once the generation process is finished, a generated MIDI file with the provided file name will have been saved in the midi sub-directory.

```
    ./bin/midi/{FILE NAME}.mid
```

This can be played as, is provided your system supports playing MIDI files or converted into an MP3 using any MIDI to MP3 converter.

## Acknowledgments

These are the documentation/guides that inspired the source code of this project.

* [Music21 User's Guide](https://web.mit.edu/music21/doc/usersGuide/usersGuide_16_tinyNotation.html)
* [PyTorch Training Docs](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
* [Relative Positional Encoding by Jake Tae](https://jaketae.github.io/study/relative-positional-encoding/)

This is the source of the datasets used in evaluation and the example Piano roll dataset:

* [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)