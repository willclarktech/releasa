
clean_training_data:
	rm -rf ./data/rgb-digits-test
	rm -rf ./data/rgb-digits-train

generate_trainingdata:
	./make_captchas.py -m 10000 -n 1000 -f -r 1.5

new: clean_training_data generate_trainingdata

train: ./train.py

test: ./test.py
