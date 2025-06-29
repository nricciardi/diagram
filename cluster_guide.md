# The Nic's Master Guide

[Documentazione ufficiale](https://ailb-web.ing.unimore.it/coldfront/documentation)

## 1. Connessione con il cluster

### 1.1 Richiedere accesso al cluster

Richiedere accesso al cluster. 

Apri un ticket su: [https://ailb-web.ing.unimore.it/tickets/](https://ailb-web.ing.unimore.it/tickets/)

### 1.2 Connessione VPN Unimore

[Qui le istruzioni d'uso ufficiali](https://www.sirs.unimore.it/site/home/servizi/accesso-vpn.html).

Per ambienti Unix-like:

1. Scaricare `openfortivpn`

```bash
sudo apt-get install openfortivpn
# or
sudo zypper install openfortivpn
# or
...
```

2. Connettersi alla rete Unimore:

```bash
openfortivpn vpn.unimore.it:443 -u <numero-nella-mail>
```

### 1.3 Connessione SSH

```bash
ssh <aimagelab-username>@ailb-login-02.ing.unimore.it
```

(Buona norma sarebbe verificare il certificato, ma noi digiteremo `yes` senza paura)

> [!TIP]
> `<aimagelab-username>` generalmente è *prima lettera del nome + cognome* (e.g. `mrossi`).


## 2. Primo accesso

Scegli se vuoi usare Anaconda o Venv.


### Configurazione Anaconda

1. Configurare l'ambiente Anaconda:

```bash
conda init
```

2. Aggiornare la shell per vedere i cambiamenti (due possibilità: lanciare il comando `bash` oppure chiudere e riaprire la sessione SSH).
3. Creare un proprio ambiente Conda

```bash
conda create --name <env-name> python=3.x
```

I tuoi dubbi saranno:

- Devo installare le cose che mi chiede? **Sì**
- Oh no un errore di spazio... cosa faccio? **Rilancia il comando**

1. Attiva l'ambiente Conda

```bash
conda activate <env-name> 
```

#### Installazione dipendenze

Questo è il buon senso comune:

> Sia `conda` che `pip` possono essere usati per installare pacchetti in un ambiente Conda, ma non sono equivalenti, e ci sono casi in cui è meglio usare uno rispetto all'altro.
> Usa `conda` install quando il pacchetto che ti serve è disponibile nei repository Conda.
> Usa `pip` install solo se il pacchetto non è disponibile su Conda oppure se hai un file `requirements.txt` non compatibile con Conda.

L'esperienza invece dice: **mai usare conda install sul cluster**, ma `pip`.

Visto che so che ti servirà, ecco un esempio con Pytorch:

```bash
pip3 install torch
```

Testa se funziona con `conda run -n arrow python -c "import torch"`

A patto che tu abbia tenuto aggiornato correttamente i requirements, meglio usare `pip3 install -r requirements.txt`

Se non ti fidi del tuo `requirements.txt`:

```bash
pip3 install torch opencv-python matplotlib requests pillow pandas torchvision numpy shapely transformers sentencepiece protobuf torchmetrics scikit-learn
```

##### Troubleshooting

Ci sono problemi nell'installazione? Nulla più semplice: rilancia il comando.

Continuano ad esserci problemi?

```bash
conda deactivate
conda env remove --name <env-name>
conda clean --all
```

Poi riparti da capo con la creazione dell'env conda.

Continui ad avere problemi? Ho la soluzione per te: mail a Baraldi.

## 3. Esecuzione script

1. Scegli il modulo Cuda in base alla tua versione Python

```bash
module avail
```

> [!NOTE]
> Per esempio, con Python 3.12 usare Cuda 12 (e.g. `cuda/12.6.3-none-none`).

2. Prepara il file bash da eseguire:

```sh
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# In teoria usare: . /usr/local/anaconda3/etc/profile.d/conda.sh

module load cuda/12.6.3-none-none

python3 <path/to/py-script.py> [parameters]
```

> [!NOTE]
> Nella doc dice di usare `. /usr/local/anaconda3/etc/profile.d/conda.sh` ma a me non funziona: dice non esiste il file. Il lato positivo è che pare funzioni lo stesso.

3. Rendilo eseguibile:

```bash
chmod u+x <train-script.sh>
```

4. Eseguilo:

```bash
sbatch <train-script.sh>
```

5. Segui l'andamento del job (e spera che stia andando):

```bash
tail -f train.out
# or
tail -f train.err
```

6. Verifica che abbia finito:

```bash
squeue -u <username>
```


### Troubleshooting

#### No module named 'src'

Classicone. Al posto di usare `sys`, una soluzione più elegante è usare la variabile di ambiente `PYTHONPATH`. Se non ti piace usa `python -m` e includi il modulo che ti serve.

Supponendo che ormai frustato tu non voglia sapere nulla di `python -m`, cambia lo script sh di esecuzione del job come segue:

```sh
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# In teoria usare: . /usr/local/anaconda3/etc/profile.d/conda.sh

module load cuda/12.6.3-none-none

export PYTHONPATH=<path/to/module>

python3 <path/to/py-script.py> [parameters]
```

> [!NOTE]
> `<path/to/module>` è la directory *di livello superiore* rispetto al modulo che non trovi: in questo caso `src`. 
> 
> Per esempio: `export PYTHONPATH=/work/cvcs2025/garagnani_napolitano_ricciardi/` se la directory (aka modulo Python) è in `/work/cvcs2025/garagnani_napolitano_ricciardi/src`


## 4. Configura VSCode

1. Installa le estensioni *Microsoft*:

- Remote - SSH
- Remote - SSH: Editing Configuration Files
- Python
- Pylance

2. Connettiti al cluster via SSH:

```
F1 da tastiera
Remote-SSH: Connect to Host
<Invio>
ssh <aimagelab-username>@ailb-login-02.ing.unimore.it
```

3. Creare file `bash.sh` nella root del progetto con all'interno:

```
srun -Q --immediate=10 -w <host> --partition=all_serial --account=<account> --gres=gpu:1 --time 60:00 --pty bash
```

dove `host` è il frontend node a cui si è connessi.

4. Assegna i permessi di esecuzione:

```bash
chmod u+x bash.sh
```

5. Aggiungere ai settings `.vscode` del progetto:

```bash
"terminal.integrated.automationProfile.linux": { "path": "$projectFolder/bash.sh" }
```

6. Restart VSCode
7. Premere il pulsante in basso a sinistra e connettersi all'host remoto
