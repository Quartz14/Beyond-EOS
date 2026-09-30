# Beyond-EOS

Overview of code to run experiments:

1. To finetune the model by adding trainable bias vectors to every layer of the model run
   ```
   python train/train_sv.py --base_model 'meta-llama/Meta-Llama-3.1-8B-Instruct' --dataset_name '' --inst '' --output_dir '' 
   ```
2. To obtain generations:
   ```
   python get_gen.py --alpha 1 --target_layers "0 1 2 3 4 5" --model_path 'meta-llama/Meta-Llama-3.1-8B-Instruct' --data_path 'path to test set' --sv_path 'path to saved checkpoint' --save_name ''
   ```
   Varying the target_layers allows control on which layers we want to add the trained steering vectors to observe corresponding changes.

3. For centroid based clustering we use a set of manually extracted examples as prototypes to ground the centroids on. This helps us automate the classification process later on for the entire dataset. The examples used are presented in file prototypes.json.
4. We use this file to perform clustering of sentences in the generations and label them as 'Rule', 'Generation' or 'Noise'. And plot the frequency of rules across checkpoints and layers to obtain the analysis of RQ1
   ```
   python RQ1_plot.py --ckpts_list '50 100 250 600' --dataset ''
   ```
5. For the experiments of RQ2 we include additonal models and datasets and run the same experiments as above. To generate the plots consolidating the data:
   ```
   python RQ2_plots.py
   ```
6. Utilizing IM to gain insights on the model's interpretation of dataset we perform clustering and corresponding visualizations.
   ```
   python RQ3_plot.py --ckpts_list '50 100 250 600' --dataset ''
   ```
7. To evaluate the context relevance of the rules in IM use the following script:
   ```
   python evaluate_IM_prometheus.py -test_path 'path to test file' --gen_path 'path to generations file'
   ```
