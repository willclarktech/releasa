# releasa

The opposite of captcha.

## generate, then solve
![montage](/montage.png)

### `make_captchas.py` Generate training set
```
./make_captchas.py -h
usage: make_captchas.py [-h] [-d] [-n N_TEST_IMAGES] [-m N_TRAIN_IMAGES] [-r RANDOM_NOISE] [-f]

make captcha training data for releasa

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           debug mode: Generate only one image. Dont save only show and exit.
  -n N_TEST_IMAGES, --n_test_images N_TEST_IMAGES
                        number of test captchas to create
  -m N_TRAIN_IMAGES, --n_train_images N_TRAIN_IMAGES
                        number of train captchas to create
  -r RANDOM_NOISE, --random-noise RANDOM_NOISE
                        add random noise to image. n >= 0; (~0.1-10)
  -f, --font-randomize  randomized fonts for each captcha
```

### generate training set with default paramaters
```
make new
```

### generate training set with custom parameters
```
./make_captchas.py --n_test_images 1000 --m_train_images 100 --font-randomize --random-noise 1.5
```

### debug mode
generate captcha and displayes but does not save to disk
```
./make_captchas.py --debug
```
