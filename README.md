# Name generator
A simple command line name generator based on Markov chains with memory with temperature sampling. 

Also allows training own model. Examples for international first names are located in `models/` directory.

## Usage
Generate a name:
```commandline
python namegen.py sample models/names_1.model --temperature=1.5
```

Train a new model:
```commandline
python namegen.py train names.txt models/names_4.model --order=4
```

## Acknowledgements
Trained international names model with data from [Dominic Tarr's random-name project](https://github.com/dominictarr/random-name) on Github.