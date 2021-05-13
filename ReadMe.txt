The overall structure:
1. Data is in the home directory.
2. Scripts: testes' scripts
   pretrain.sh: setting up cuda environment, sending what GPU cards are  assigned; call     
   run.py
3. run.py: Is called by shell scripts (e.g., pretrain.sh) to run experiments.
   Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py
4. experiments/pretrain/emain.py: is called by run.py. Is just the main train loop.  The   
   main thing is setup to get all the model components.   
5. lib/utils: the training utils. images.py: for visualization. 
    statistics.py: track the experiments using weights and biases API. WandBTracker object  
    is initialized at the beginning of the training and and is updated during the training    
    and connected to the weights and biases database.