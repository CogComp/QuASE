// BERT embeddings + Standard QALM embeddings for Coref
{
  "dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-cased",
          "do_lowercase": false,
          "use_starting_offsets": true,
          "truncate_long_sequences": false
      },
      "semanticbert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "do_lowercase": true,
          "use_starting_offsets": true,
          "truncate_long_sequences": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      }
    },
    "max_span_width": 10
  },
  "train_data_path": "/path/to/data/Coref/train.english.v4_gold_conll",
  "validation_data_path": "/path/to/data/Coref/dev.english.v4_gold_conll",
  "test_data_path": "/path/to/data/Coref/test.english.v4_gold_conll",
  "model": {
    "type": "coref",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
            "semanticbert": ["semanticbert", "semanticbert-offsets"],
            "token_characters": ["token_characters"]
      },
      "token_embedders": {
         "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased"
         },
         "semanticbert": {
                "type": "semanticbert-pretrained",
                "pretrained_model": "/path/to/models/QAMR/outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin",
                "top_layer_only": false
         },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [5]
            }
        }
      }
    },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 768+100+768,
        "hidden_size": 200,
        "num_layers": 1
    },
    "mention_feedforward": {
        "input_dim": 2456,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": 7388,
        "num_layers": 2,
        "hidden_dims": 150,
        "activations": "relu",
        "dropout": 0.2
    },
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ],
    "lexical_dropout": 0.5,
    "feature_size": 20,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    "max_antecedents": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1
  },
  "trainer": {
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam"
    }
  }
}