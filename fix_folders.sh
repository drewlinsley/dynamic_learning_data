git clone https://huggingface.co/datasets/YWjimmy/PeRFception-v1-1
git clone https://huggingface.co/datasets/YWjimmy/PeRFception-v1-2
git clone https://huggingface.co/datasets/YWjimmy/PeRFception-v1-3
mkdir PeRFception-models




cd PeRFception-v1-2
mv */* .
cd ../PeRFception-v1-1
mv */* .
cd PeRFception-v1-3
mv */* .

mkdir PeRFception-models
mv PeRFception-v1-1/*/* PeRFception-models/
mv PeRFception-v1-2/*/* PeRFception-models/
mv PeRFception-v1-3/*/* PeRFception-models/

