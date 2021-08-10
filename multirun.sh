python run_seq2vec.py -c experiments/simplest_span_class.json -m /mnt/sda/iReC/simplest_span_classifier
wait 
python run_seq2vec.py -c experiments/s2v_con_mask.json -m /mnt/sda/iReC/s2v_con_mask
wait 
python run_seq2vec.py -c experiments/s2v_con_cls.json -m /mnt/sda/iReC/s2v_con_cls
wait 
python run_seq2vec.py -c experiments/s2v_con_pad.json -m /mnt/sda/iReC/s2v_con_pad
wait
python run_seq2vec.py -c experiments/s2v_con_sep.json -m /mnt/sda/iReC/s2v_con_sep
wait
python run_seq2vec.py -c experiments/s2v_con_zero.json -m /mnt/sda/iReC/s2v_con_zero
wait
python run_seq2vec.py -c experiments/s2v_noc_cls.json -m /mnt/sda/iReC/s2v_noc_cls
wait 
python run_seq2vec.py -c experiments/s2v_noc_pad.json -m /mnt/sda/iReC/s2v_noc_pad
wait
python run_seq2vec.py -c experiments/s2v_noc_sep.json -m /mnt/sda/iReC/s2v_noc_sep
wait
python run_seq2vec.py -c experiments/s2v_noc_zero.json -m /mnt/sda/iReC/s2v_noc_zero