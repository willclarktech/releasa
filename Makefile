
clean_training_data:
	rm -rf ./data/rgb-digits-test
	rm -rf ./data/rgb-digits-train

generate_trainingdata:
	./make_captchas.py -n 10 -m 10 -f -r 1.5

new: clean_training_data generate_trainingdata
