version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task train {
    input {
        File ML_config
        File data_train
        File data_test
        File ckpt
        File ground_truth 
        File credentials_json
        String git_repo
        String git_branch_or_commit
        String main_file_name 
    }


    command <<<
        echo "CAZZO"       
 
        exec_dir=$(pwd)
        echo "--> $exec_dir"
        echo "START --> Content of exectution dir"
        echo $(ls)
        
        
        # 2. clone the repository in the checkout_dir
        # for public repository use:
        git clone ~{git_repo} ./checkout_dir
        # for private repository use:
        # github_token=$(cat ~{credentials_json} | grep -o '"GITHUB_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$' | sed 's/"//g')
        # git_repo_with_token=$(echo ~{git_repo} | sed "s/github/$github_token@github/")
        # git clone $git_repo_with_token ./checkout_dir 

        # 3. checkout the branch
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        echo "AFTER GIT --> Content of checkout dir"
        echo $(ls)
        
        # 4. link the file which have been localized to checkout_dir/src
        # and give them the name the main.py expects
        ln -s ~{ML_config} ./src/config.yaml
        ln -s ~{data_train} ./src/data_train.pt
        ln -s ~{data_test} ./src/data_test.pt
        ln -s ~{ckpt} ./src/ckpt.pt
        ln -s ~{ground_truth} ./src/ground_truth.pt
        echo "AFTER CHANGING NAMES --> Content of checkout dir"
        echo $(ls)

        # 5. run python code only if NEPTUNE credentials are found
        neptune_token=$(cat ~{credentials_json} | grep -o '"NEPTUNE_API_TOKEN"\s*:\s*"[^"]*"' | grep -o '"[^"]*"$')
        if [ ! -z $neptune_token ]; then
           export NEPTUNE_API_TOKEN=$neptune_token
           cd ./src 
           python ~{main_file_name}
        fi
    >>>
    
#    runtime {
#          docker: "python"
#          cpu: 1
#          preemptible: 3
#    }
    
    runtime {
         docker: "us.gcr.io/broad-dsde-methods/genus:latest"
         bootDiskSizeGb: 100
         memory: "26G"
         cpu: 4
         zones: "us-east1-d us-east1-c"
         gpuCount: 1
         gpuType:  "nvidia-tesla-t4" #"nvidia-tesla-p100" #"nvidia-tesla-k80"
         maxRetries: 0
         preemptible_tries: 0
    }

}

workflow neptune_ml {

    input {
        File ML_config 
        File data_train 
        File data_test
        File ckpt
        File ground_truth 
        File credentials_json 
        String git_repo
        String git_branch_or_commit 
        String main_file_name 
    }

    call train { 
        input :
            ML_config = ML_config,
            credentials_json = credentials_json,
            data_train = data_train,
            data_test = data_test,
            ckpt = ckpt,
            ground_truth = ground_truth, 
            git_repo = git_repo,
            git_branch_or_commit = git_branch_or_commit,
            main_file_name = main_file_name 
    }
}
