# TODO: create shell script for Problem 1
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rbH1aZJBrWw_aGvDCafLhMserm_Skaxg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rbH1aZJBrWw_aGvDCafLhMserm_Skaxg" -O FC_best.pth.tar && rm -rf /tmp/cookies.txt

python3 src/hw4_p1.py --video_path $1 --csv_path $2 --output_csv $3
