// Elmo embeddings + Standard QALM embeddings for SRL
{
    "dataset_reader": {
        "type": "srl",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
             "semanticbert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": true
             }
        }
    },
    "train_data_path": "/path/to/data/conll-formatted-ontonotes-5.0/data/train",
    "validation_data_path": "/path/to/data/conll-formatted-ontonotes-5.0/data/development",
    "test_data_path": "/path/to/data/conll-formatted-ontonotes-5.0/data/test",
    "model": {
        "type": "srl",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "elmo": ["elmo"],
                "semanticbert": ["semanticbert", "semanticbert-offsets"]
            },
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.1
                },
                "semanticbert": {
                    "type": "semanticbert-pretrained",
                    "pretrained_model": "/path/to/models/QAMR/outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin",
                    "top_layer_only": false
                }
            }
        },
        "initializer": [
            [
                "tag_projection_layer.*weight",
                {
                    "type": "orthogonal"
                }
            ]
        ],
        // NOTE: This configuration is correct, but slow.
        "encoder": {
            "type": "alternating_lstm",
            "input_size": 1024 + 100 + 768,
            "hidden_size": 300,
            "num_layers": 8,
            "recurrent_dropout_probability": 0.1,
            "use_input_projection_bias": false
        },
        "binary_feature_dim": 100,
        "regularizer": [
            [
                ".*scalar_parameters.*",
                {
                    "type": "l2",
                    "alpha": 0.001
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ],
        "batch_size": 80
    },
    "trainer": {
        "num_epochs": 500,
        "grad_clipping": 1.0,
        "patience": 200,
        "num_serialized_models_to_keep": 10,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": [0,1],
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        }
    }
}