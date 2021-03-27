import os
import torch
from transformers import BertModel
from src.config.query_rewriter_config import ROBERTA_WWM_EXT, BART_WWM_EXT
from src.model.BART_origin import BartForConditionalGeneration, BartConfig


def convert_weignt_bert_to_bart(bart_model_output_path):
    bert_model = BertModel.from_pretrained(ROBERTA_WWM_EXT)
    config_bart = BartConfig.from_pretrained('../../data/pretrained_model/roberta/bart_config.json')
    config_bart.extra_pos_embeddings = 0
    bart_model = BartForConditionalGeneration(config=config_bart)
    with torch.no_grad():
        bert_named_parameters = {k: v for k, v in bert_model.named_parameters()}
        for name, param in bart_model.named_parameters():
            if "decoder" in name:
                continue
            if name == "model.shared.weight":
                bert_word_embedding_weight = bert_named_parameters["embeddings.word_embeddings.weight"]
                param.copy_(bert_word_embedding_weight)
            elif name == "model.encoder.token_type_embeddings.weight":
                bert_token_type_embeddings_weight = bert_named_parameters["embeddings.token_type_embeddings.weight"]
                param.copy_(bert_token_type_embeddings_weight)
            elif name == "model.encoder.embed_positions.weight":
                bert_position_embeddings_weight = bert_named_parameters["embeddings.position_embeddings.weight"]
                param.copy_(bert_position_embeddings_weight)
            elif name == "model.encoder.layernorm_embedding.weight":
                bert_embeddings_LayerNorm_weight = bert_named_parameters["embeddings.LayerNorm.weight"]
                param.copy_(bert_embeddings_LayerNorm_weight)
            elif name == "model.encoder.layernorm_embedding.bias":
                bert_embeddings_LayerNorm_bias = bert_named_parameters["embeddings.LayerNorm.bias"]
                param.copy_(bert_embeddings_LayerNorm_bias)

            elif "self_attn.q_proj.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_query_weight = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.query.weight"]
                param.copy_(bert_encoder_layer_N_attention_self_query_weight)
            elif "self_attn.q_proj.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_query_bias = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.query.bias"]
                param.copy_(bert_encoder_layer_N_attention_self_query_bias)
            elif "self_attn.k_proj.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_key_weight = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.key.weight"]
                param.copy_(bert_encoder_layer_N_attention_self_key_weight)
            elif "self_attn.k_proj.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_key_bias = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.key.bias"]
                param.copy_(bert_encoder_layer_N_attention_self_key_bias)
            elif "self_attn.v_proj.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_value_weight = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.value.weight"]
                param.copy_(bert_encoder_layer_N_attention_self_value_weight)
            elif "self_attn.v_proj.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_self_value_bias = bert_named_parameters[f"encoder.layer.{layer_id}.attention.self.value.bias"]
                param.copy_(bert_encoder_layer_N_attention_self_value_bias)

            elif "self_attn.out_proj.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_output_dense_weight = bert_named_parameters[f"encoder.layer.{layer_id}.attention.output.dense.weight"]
                param.copy_(bert_encoder_layer_N_attention_output_dense_weight)
            elif "self_attn.out_proj.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_output_dense_bias = bert_named_parameters[f"encoder.layer.{layer_id}.attention.output.dense.bias"]
                param.copy_(bert_encoder_layer_N_attention_output_dense_bias)
            elif "self_attn_layer_norm.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_output_LayerNorm_weight = bert_named_parameters[f"encoder.layer.{layer_id}.attention.output.LayerNorm.weight"]
                param.copy_(bert_encoder_layer_N_attention_output_LayerNorm_weight)
            elif "self_attn_layer_norm.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_attention_output_LayerNorm_bias = bert_named_parameters[f"encoder.layer.{layer_id}.attention.output.LayerNorm.bias"]
                param.copy_(bert_encoder_layer_N_attention_output_LayerNorm_bias)

            elif "fc1.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_intermediate_dense_weight = bert_named_parameters[f"encoder.layer.{layer_id}.intermediate.dense.weight"]
                param.copy_(bert_encoder_layer_N_intermediate_dense_weight)
            elif "fc1.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_intermediate_dense_bias = bert_named_parameters[f"encoder.layer.{layer_id}.intermediate.dense.bias"]
                param.copy_(bert_encoder_layer_N_intermediate_dense_bias)
            elif "fc3.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_output_dense_weight = bert_named_parameters[f"encoder.layer.{layer_id}.output.dense.weight"]
                param.copy_(bert_encoder_layer_N_output_dense_weight)
            elif "fc3.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_output_dense_bias = bert_named_parameters[f"encoder.layer.{layer_id}.output.dense.bias"]
                param.copy_(bert_encoder_layer_N_output_dense_bias)
            elif "final_layer_norm.weight" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_output_LayerNorm_weight = bert_named_parameters[f"encoder.layer.{layer_id}.output.LayerNorm.weight"]
                param.copy_(bert_encoder_layer_N_output_LayerNorm_weight)
            elif "final_layer_norm.bias" in name:
                layer_id = name.split(".")[3]
                bert_encoder_layer_N_output_LayerNorm_bias = bert_named_parameters[f"encoder.layer.{layer_id}.output.LayerNorm.bias"]
                param.copy_(bert_encoder_layer_N_output_LayerNorm_bias)
        if not os.path.exists(bart_model_output_path):
            os.mkdir(bart_model_output_path)
        bart_model.save_pretrained(bart_model_output_path)


if __name__ == "__main__":
    convert_weignt_bert_to_bart(bart_model_output_path=BART_WWM_EXT)
    pass
