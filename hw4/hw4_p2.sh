# TODO: create shell script for Problem 2
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tonVSIUddxxKE0usvA1k7r6_CdgCBB5F' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tonVSIUddxxKE0usvA1k7r6_CdgCBB5F" -O BiRNN_best.pth.tar && rm -rf /tmp/cookies.txt
python3 src/hw4_p2.py --video_path $1 --csv_path $2 --output_csv $3

