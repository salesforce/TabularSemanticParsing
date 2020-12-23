#!/usr/bin/env bash

export PYTHONPATH="`pwd`;$PYTHONPATH"

source $1
exp=$2
gpu=$3
ARGS=${@:4}

question_split_flag=''
if [[ $question_split = *"True"* ]]; then
    question_split_flag="--question_split"
fi
query_split_flag=''
if [[ $query_split = *"True"* ]]; then
    query_split_flag="--query_split"
fi
question_only_flag=''
if [[ $question_only = *"True"* ]]; then
    question_only_flag="--question_only"
fi
normalize_variables_flag=''
if [[ $normalize_variables = *"True"* ]]; then
    normalize_variables_flag="--normalize_variables"
fi
share_vocab_flag=''
if [[ $share_vocab = *"True"* ]]; then
    share_vocab_flag="--share_vocab"
fi
denormalize_sql_flag=''
if [[ $denormalize_sql = *"True"* ]]; then
    denormalize_sql_flag="--denormalize_sql"
fi
omit_fram_clause_flag=''
if [[ $omit_from_clause = *"True"* ]]; then
    omit_from_clause_flag="--omit_from_clause"
fi
no_join_condition_flag=''
if [[ $no_join_condition = *"True"* ]]; then
    no_join_condition_flag="--no_join_condition"
fi
table_shuffling_flag=''
if [[ $table_shuffling = *"True"* ]]; then
    table_shuffling_flag="--table_shuffling"
fi
use_lstm_encoder_flag=''
if [[ $use_lstm_encoder = *"True"* ]]; then
    use_lstm_encoder_flag="--use_lstm_encoder"
fi
use_meta_data_encoding_flag=''
if [[ $use_meta_data_encoding = *"True"* ]]; then
    use_meta_data_encoding_flag="--use_meta_data_encoding"
fi
use_graph_encoding_flag=''
if [[ $use_graph_encoding = *"True"* ]]; then
    use_graph_encoding_flag="--use_graph_encoding"
fi
sql_consistency_check_flag=''
if [[ $sql_consistency_check = *"True"* ]]; then
    sql_consistency_check_flag="--sql_consistency_check"
fi
use_typed_field_markers_flag=''
if [[ $use_typed_field_markers = *"True"* ]]; then
    use_typed_field_markers_flag="--use_typed_field_markers"
fi
read_picklist_flag=''
if [[ $read_picklist = *"True"* ]]; then
    read_picklist_flag="--read_picklist"
fi 
use_picklist_flag=''
if [[ $use_picklist = *"True"* ]]; then
    use_picklist_flag="--use_picklist"
fi
no_anchor_text_flag=''
if [[ $no_anchor_text = *"True"* ]]; then
    no_anchor_text_flag="--no_anchor_text"
fi
process_sql_in_execution_order_flag=''
if [[ $process_sql_in_execution_order = *"True"* ]]; then
    process_sql_in_execution_order_flag="--process_sql_in_execution_order"
fi
sample_ground_truth_flag=''
if [[ $sample_ground_truth = *"True"* ]]; then
    sample_ground_truth_flag="--sample_ground_truth"
fi
save_nn_weights_for_visualizations_flag=''
if [[ $save_nn_weights_for_visualizations = *"True"* ]]; then
    save_nn_weights_for_visualizations_flag="--save_nn_weights_for_visualizations"
fi
fix_pretrained_transformer_parameters_flag=''
if [[ $fix_pretrained_transformer_parameters = *"True"* ]]; then
    fix_pretrained_transformer_parameters_flag="--fix_pretrained_transformer_parameters"
fi
use_oracle_tables_flag=''
if [[ $use_oracle_tables = *"True"* ]]; then
    use_oracle_tables_flag="--use_oracle_tables"
fi
atomic_value_copy_flag=''
if [[ $atomic_value_copy = *"True"* ]]; then
    atomic_value_copy_flag='--atomic_value_copy'
fi
use_additive_features_flag=''
if [[ $use_additive_features = *"True"* ]]; then
    use_additive_features_flag="--use_additive_features"
fi
data_parallel_flag=''
if [[ $data_parallel = *"True"* ]]; then
    data_parallel_flag="--data_parallel"
fi
save_best_model_only_flag=''
if [[ $save_best_model_only = *"True"* ]]; then
    save_best_model_only_flag="--save_best_model_only"
fi
augment_with_wikisql_flag=''
if [[ $augment_with_wikisql = *"True"* ]]; then
    augment_with_wikisql_flag="--augment_with_wikisql"
fi
random_field_order_flag=''
if [[ $random_field_order = *"True"* ]]; then
    random_field_order_flag="--random_field_order"
fi

cmd="python3 -m src.experiments \
    $exp \
    --data_dir $data_dir \
    --db_dir $db_dir \
    --dataset_name $dataset_name \
    $question_split_flag \
    $query_split_flag \
    $question_only_flag \
    $normalize_variables_flag \
    $share_vocab_flag \
    $denormalize_sql_flag \
    $omit_from_clause_flag \
    $no_join_condition_flag \
    $table_shuffling_flag \
    $use_lstm_encoder_flag \
    $use_meta_data_encoding_flag \
    $use_graph_encoding_flag \
    $sql_consistency_check_flag \
    $use_typed_field_markers_flag \
    $use_picklist_flag \
    --anchor_text_match_threshold $anchor_text_match_threshold \
    $no_anchor_text_flag \
    $read_picklist_flag \
    --top_k_picklist_matches $top_k_picklist_matches \
    $process_sql_in_execution_order_flag \
    $sample_ground_truth_flag \
    $use_oracle_tables_flag \
    --num_random_tables_added $num_random_tables_added \
    $atomic_value_copy_flag \
    $use_additive_features_flag \
    $save_nn_weights_for_visualizations_flag \
    $data_parallel_flag \
    $save_best_model_only_flag \
    --schema_augmentation_factor $schema_augmentation_factor \
    $random_field_order_flag \
    --data_augmentation_factor $data_augmentation_factor \
    $augment_with_wikisql_flag \
    --vocab_min_freq $vocab_min_freq \
    --text_vocab_min_freq $text_vocab_min_freq \
    --program_vocab_min_freq $program_vocab_min_freq \
    --num_values_per_field $num_values_per_field \
    --max_in_seq_len $max_in_seq_len \
    --max_out_seq_len $max_out_seq_len \
    --model $model \
    --num_steps $num_steps \
    --curriculum_interval $curriculum_interval \
    --num_peek_steps $num_peek_steps \
    --num_accumulation_steps $num_accumulation_steps \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --encoder_input_dim $encoder_input_dim \
    --encoder_hidden_dim $encoder_hidden_dim \
    --decoder_input_dim $decoder_input_dim \
    --num_rnn_layers $num_rnn_layers \
    --num_const_attn_layers $num_const_attn_layers \
    --emb_dropout_rate $emb_dropout_rate \
    --pretrained_lm_dropout_rate $pretrained_lm_dropout_rate \
    --rnn_layer_dropout_rate $rnn_layer_dropout_rate \
    --rnn_weight_dropout_rate $rnn_weight_dropout_rate \
    --cross_attn_dropout_rate $cross_attn_dropout_rate \
    --cross_attn_num_heads $cross_attn_num_heads \
    --res_input_dropout_rate $res_input_dropout_rate \
    --res_layer_dropout_rate $res_layer_dropout_rate \
    --ff_input_dropout_rate $ff_input_dropout_rate \
    --ff_hidden_dropout_rate $ff_hidden_dropout_rate \
    --pretrained_transformer $pretrained_transformer \
    $fix_pretrained_transformer_parameters_flag \
    --bert_finetune_rate $bert_finetune_rate \
    --learning_rate $learning_rate \
    --learning_rate_scheduler $learning_rate_scheduler \
    --trans_learning_rate_scheduler $trans_learning_rate_scheduler \
    --warmup_init_lr $warmup_init_lr \
    --warmup_init_ft_lr $warmup_init_ft_lr \
    --num_warmup_steps $num_warmup_steps \
    --grad_norm $grad_norm \
    --decoding_algorithm $decoding_algorithm \
    --beam_size $beam_size \
    --bs_alpha $bs_alpha \
    --gpu $gpu \
    $ARGS"

echo "run $cmd"
$cmd
