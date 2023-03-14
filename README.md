# range3/pegasus-lm-slackbot
```bash
git clone https://github.com/range3/pegasus-lm.git ../
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
pip install -e ../pegasus-lm
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu117
pip install git+https://github.com/huggingface/transformers
```
