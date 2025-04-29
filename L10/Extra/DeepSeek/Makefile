F=./install.sh

all:
	@ # NOTE: https://www.analyticsvidhya.com/blog/2025/01/run-deepseek-models-locally/
	@ test -f $F || curl -fsSL https://ollama.com/install.sh > $F 
	@ chmod ugo+x $F
	@ test -f /usr/local/bin/ollama || ./install.sh
	@ systemctl start ollama
	@ ollama list | grep "deepseek-r1:1.5" >/dev/null || ollama pull deepseek-r1:1.5b
	@ ollama run deepseek-r1:1.5b
	@ echo DONE
