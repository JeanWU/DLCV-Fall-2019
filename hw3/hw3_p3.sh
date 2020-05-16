# TODO: download models
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VZ1Cg713ufD5YOL3jpj-zadDCpKCIYdu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VZ1Cg713ufD5YOL3jpj-zadDCpKCIYdu" -O m2s_dann_feature_extractor.pth.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QbWwYUdTFw8NhdNC-eIkPTjSAy4hK8x0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QbWwYUdTFw8NhdNC-eIkPTjSAy4hK8x0" -O m2s_dann_label_predictor.pth.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Rt4CYLxUZo6ICLfBvBERgX3oRB0hwZ58' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Rt4CYLxUZo6ICLfBvBERgX3oRB0hwZ58" -O s2m_dann_feature_extractor.pth.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nJHh56t9hXLtWESF44hu1hIPAoPlwB1s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nJHh56t9hXLtWESF44hu1hIPAoPlwB1s" -O s2m_dann_label_predictor.pth.tar && rm -rf /tmp/cookies.txt
python3 src/hw3_p3.py --data_dir $1 --target_domain $2 --save_folder $3 --image_size 28