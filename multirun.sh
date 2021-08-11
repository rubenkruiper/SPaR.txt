python run_tagger.py -c experiments/dr.00_enc_1_h192_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/00_enc_1_h192_ffnn_1_h200_1e
wait 
python run_tagger.py -c experiments/dr.00_enc_1_h384_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.00_enc_1_h384_ffnn_1_h200_1e-4_32
wait 
python run_tagger.py -c experiments/dr.01_enc_1_h384_ffnn_1_h200_1e-3_32.json -m /mnt/sda/mwe/dr.01_enc_1_h384_ffnn_1_h200_1e-3_32
wait 
python run_tagger.py -c experiments/dr.01_enc_1_h384_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_1_h384_ffnn_1_h200_1e-4_32
wait
python run_tagger.py -c experiments/dr.01_enc_1_h384_no_ffnn_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_1_h384_no_ffnn_1e-4_32
wait
python run_tagger.py -c experiments/dr.01_enc_1_h786_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_1_h786_ffnn_1_h200_1e-4_32
wait
python run_tagger.py -c experiments/dr.01_enc_1_h786_ffnn_2_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_1_h786_ffnn_2_h200_1e-4_32
wait 
python run_tagger.py -c experiments/dr.01_enc_2_h192_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_2_h192_ffnn_1_h200_1e-4_32
wait
python run_tagger.py -c experiments/dr.01_enc_2_h384_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_2_h384_ffnn_1_h200_1e-4_32
wait
python run_tagger.py -c experiments/dr.01_enc_passthrough_ffnn_1_h200_1e-4_32.json -m /mnt/sda/mwe/dr.01_enc_passthrough_ffnn_1_h200_1e-4_32