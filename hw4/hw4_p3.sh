# TODO: create shell script for Problem 3
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wgZZ2_zwEgJx3JoTx6DiDysEwN94FRa-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wgZZ2_zwEgJx3JoTx6DiDysEwN94FRa-" -O seq2seq_best.pth.tar && rm -rf /tmp/cookies.txt
python3 src/hw4_p3.py --full_video_path $1 --output_csv $2
