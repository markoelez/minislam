
.PHONY all
all: pangolin install

.PHONY pangolin
pangolin: ./install_pangolin.sh

.PHONY install
install: requirements.txt
	pip install -r requirements.txt
