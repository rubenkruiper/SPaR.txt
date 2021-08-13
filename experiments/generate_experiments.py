import itertools, json

def generate_config_dict(num_epochs: int,
                         dropout: float,
                         encoder_layers: int,
                         encoder_hidden: int,
                         ffnn_layers: int,
                         ffnn_hidden: int,
                         learning_rate: float,
                         batch_size: int,
                         train_path: str = "data/train/",
                         val_path: str = "data/val/",
                         bert_model_name: str = "SpanBERT/spanbert-base-cased",
                         encoder_bidirectional: bool = True):

    experiment_name = "dr{}_enc_{}_h{}_ffnn_{}_h{}_{}_{}".format(dropout,
                                                                 encoder_layers,
                                                                 encoder_hidden,
                                                                 ffnn_layers,
                                                                 ffnn_hidden,
                                                                 learning_rate,
                                                                 batch_size)

    if encoder_bidirectional:
        ffnn_in = 2 * encoder_hidden
    else:
        ffnn_in = encoder_hidden

    dataset_reader = {
                      "type": "tag_reader",
                      "bert_model_name": bert_model_name,
                      "token_indexers": {
                          "tokens": {
                            "type": "pretrained_transformer",
                            "model_name": bert_model_name,
                            "max_length": 512
                          }
                        }
                      }

    data_loader = {"batch_sampler": {
                      "type": "bucket",
                      "sorting_keys": ["tokens"],
                      "batch_size": batch_size
                        }}

    encoder = {
                "type": "gru",
                "input_size": 768,   # transformer hidden dim
                "hidden_size": encoder_hidden,  # any type of dim you want ?
                "num_layers": encoder_layers,
                "bidirectional": encoder_bidirectional
            }

    feedforward = {
                   "input_dim": ffnn_in,
                   "num_layers": ffnn_layers,
                   "hidden_dims": ffnn_hidden,
                   "activations": "relu",
                   "dropout": dropout
                   }

    text_field_embedder = {"token_embedders": {
                                "tokens": {
                                  "type": "pretrained_transformer",
                                  "train_parameters": False,
                                  "model_name": bert_model_name,
                                  "max_length": 512
                                }}}

    model = {"type": "my_tagger",
             "dropout": dropout,
             "encoder": encoder,
             "feedforward": feedforward,
             "include_start_end_transitions": False,
             "label_encoding": "DiscontiguousTest",
             "text_field_embedder": text_field_embedder,
             "verbose_metrics" : False
             }

    optimizer = {"type": "huggingface_adamw",
              "lr": learning_rate,
              "weight_decay": 0.1,
              "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0.05}]],
              "eps": 1e-8,
              "correct_bias": True}

    learning_rate_scheduler = {"type": "slanted_triangular"}

    callbacks = {"wandb": {
                           "summary_interval": 1,
                           "distribution_interval": 1,
                           "batch_size_interval": 1,
                           "should_log_parameter_statistics": True,
                           "should_log_learning_rate": True,
                           "project": "MWE-tagger",
                           "entity": "rkruiper",
                           "name": experiment_name,
                           "notes": experiment_name
                           }}

    trainer = {"optimizer": optimizer,
               "learning_rate_scheduler": learning_rate_scheduler,
               "callbacks": callbacks,
               "grad_norm": 1.0,
               "cuda_device": 0,
               "num_epochs": num_epochs,
               "num_serialized_models_to_keep": 1,
               "validation_metric": "+f1-measure-overall",
               "patience": 20
               }

    config_dict = {
                    "dataset_reader": dataset_reader,
                    "data_loader": data_loader,
                    "train_data_path": train_path,
                    "validation_data_path": val_path,
                    "model": model,
                    "trainer": trainer
                   }
    return config_dict, experiment_name


if __name__ == "__main__":

    num_epochs = [50]
    learning_rates = [1e-3, 5e-4]

    batch_sizes = [16, 32]

    # encoder
    dropouts = [.01, 0.05]
    encoder_hiddens = [192, 384, 768]
    encoder_layers = [1, 2]

    # ffnn
    ffnn_hiddens = [100, 200, 384]
    ffnn_layers = [1, 2]

    # Create combinations of the different values in the lists above
    for experiment in itertools.product(num_epochs, dropouts, encoder_layers, encoder_hiddens,
                                        ffnn_layers, ffnn_hiddens, learning_rates, batch_sizes):
        e, dr, el, eh, fl, fh, lr, bs = experiment
        config, name = generate_config_dict(e, dr, el, eh, fl, fh, lr, bs)
        with open(name + '.json', 'w') as f:
            json.dump(config, f)

