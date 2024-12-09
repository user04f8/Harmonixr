VENV = harmonyenv

install:
	python3 -m venv harmonyenv
	./harmonyenv/bin/pip install -r requirements.txt

preprocess:
	./harmonyenv/bin/python3 preprocessing.py

train:
	./harmonyenv/bin/python3 train.py

clean:
	rm -rf harmonyenv

reinstall: clean install