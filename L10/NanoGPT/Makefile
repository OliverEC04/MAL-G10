#!/usr/bin/env make -f

PYTHON=/opt/anaconda-2024.02/bin/python3

#DATASET=shakespeare_char
DATASET=hca
#DATASET=dr
DATADIR=./data

.PHONY:help
help:
	@ echo ; echo "RUN: make <install> | <prepare> | <train> | <pred> | <clean>"
	@ echo ; echo "  MODES: "
	@ echo "    install: python install setup"
	@ echo "    prepare: data preperation"
	@ echo "    train  : train the model"
	@ echo "    pred   : do some prediction using the trained model"
	@ echo "    clean  : clean all temp files, including trained model"
	@ echo ; echo "  DATASETS:"
	@ echo "    hca             : H.C. Andersens text,"
	@ echo "    dr              : DR news feed text,"
	@ echo "    shakespeare_char: shakespeare poems, char-mode,"
	@ echo "    openweb         : internet scraped data ala GPT-2, not possible"
	@ echo "                      without more hardware, training time in weeks!"
	@ echo "    your-own-data   : 1) create datadir, say 'mkdir ./data/mydataset',"
	@ echo "                      2) add input.txt and prepare.py in datadir"
	@ echo "                         (copy prepare.py from this dir),"
	@ echo "                      3) add config/train_mydataset.py"
	@ echo "                         (just copy from train_hca.py),"
	@ echo "                      4) modify dataset variable in the train file,"
	@ echo "                         (edit the line: dataset = 'mydataset' "
	@ echo "                          in the config/train_mydataset.py file)"
	@ echo "                      5) set DATASET=mydataset in Makefile,"
	@ echo "                      6) run 'make prepare', 'make train' etc.."
	@ echo "  HISTORY:"
	@ echo "    2024-04-09: CEF, initial version."
	@ echo "    2024-04-16: CEF, fixed DATADIR and triton version errors."
	@ echo "    2024-04-21: CEF, modified and prepared for GAN workshop."
	@ echo "    2024-05-01: CEF, more preps for GAN workshop."
	@ echo "    2024-10-01: CEF, changed to 'ATU Fordybelsesseminar' workshop."
	@ echo ; echo "  AUTHOR / LICENSE:"
	@ echo "    cef@ece.au.dk / GNU General Public License v3.0"
	@ echo

.PHONY:train
train:
	@# /home/shared/gpt_node_ok.sh
	$(PYTHON) train.py config/train_$(DATASET).py --max_iters=10 --warmup_iters=1 --eval_iters=1 --eval_interval=1
	@ $(MAKE) -s precondition_predict
	@# more parameters --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4

.PHONY:pred
pred: precondition_predict
	$(PYTHON) sample.py --out_dir=out-$(DATASET) --temperature=0.8 --num_samples=2 --start="Der kom en solda"

.PHONY:predict
predict: pred # just an alias

.PHONY:train_slurm
train_slurm:
	sbatch ./train_slurm.sh DATASET=$(DATASET)
	squeue

.PHONY:precondition_predict
precondition_predict:
	@ test -f out-$(DATASET)/ckpt.pt || echo "WARNING: missing output checkpoint 'out-$(DATASET)/ckpt.pt' file, you will not be able to make predictions!"

.PHONY:prepare
prepare:
	@ test -d $(DATADIR)      || mkdir $(DATADIR)
	@ test -d data/$(DATASET) || mkdir data/$(DATASET)
	@ test -f data/$(DATASET)/input.txt  || (echo "UNPACK   data/$(DATASET)/input.txt.."  && zcat Txt/$(DATASET).txt.gz >data/$(DATASET)/input.txt)
	@ test -f data/$(DATASET)/prepare.py || (echo "CREATING data/$(DATASET)/prepare.py.." && cp prepare.py data/$(DATASET))
	@ test -f data/$(DATASET)/train.bin  || (echo "RUNNING  data/$(DATASET)/prepare.py.." && cd data/$(DATASET) && $(PYTHON) ./prepare.py)
	@ echo PREPARE: all ok!

.PHONY:install
install:
	@echo "NOTE, CEF: using preinstalled Anaconda 2024.02,"
	@echo "      your current python is ${PYTHON}.."
	@`dirname $(PYTHON)`/pip install triton==2.2.0 # CEF: v 2.2.0 works, 2.3.0 does not!
	@# echo "WARNING: untested prepare/install script, may also need 'pip install triton'..."
	@# conda install -y python=3.10 numpy
	@# conda install -y -c conda-forge mamba micromamba
	@# mamba install -y -c pytorch -c nvidia pytorch pytorch-cuda #pytorch-cuda=11.8
	@# mamba install -y -c conda-forge datasets tqdm transformers tiktoken wandb

.PHONY:pollgpu
pollgpu:
	@while [ True ];\
	do \
		echo "\n\n\n"; \
		nvidia-smi | sed -r '/^$$/d';  gpustat; sleep 10; \
	done

.PHONY:tar
tar:
	cd .. && tar czf nanogpt.tgz --exclude=data --exclude=Slides* --exclude=*~ --exclude=.ipynb_checkpoints --exclude=__pycache__ --exclude=out-* --exclude=ckpt.pt  NanoGPT/*

.PHONY:clean
clean:
	@ rm -rf out-* slurm-*.out done*.txt __pycache__
