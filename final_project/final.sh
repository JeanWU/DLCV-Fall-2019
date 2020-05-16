# TODO: create shell script for final project
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Rrl3pje1DfhRhLK8xriCzTcOLqXUpXnr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Rrl3pje1DfhRhLK8xriCzTcOLqXUpXnr" -O unet_89.pth && rm -rf /tmp/cookies.txt

python3 src/test.py --test_dir $1 --save_dir $2 --resume_folder unet_89.pth
