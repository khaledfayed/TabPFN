from deeppipe_api.deeppipe import load_data, openml, DeepPipe

def deep_pipe_accuracy():
    
    task = openml.tasks.get_task(task_id=31)
    X_train, X_test, y_train, y_test = load_data(task, fold=0)
    deep_pipe = DeepPipe(n_iters = 50,  #bo iterations
                        time_limit = 3600, #in seconds
                        apply_cv = True,
                        create_ensemble = False,
                        ensemble_size = 10,
                        )
    deep_pipe.fit(X_train, y_train)
    score = deep_pipe.score(X_test, y_test)
    print("Test acc.:", score) 

def main():

    deep_pipe_accuracy()

if __name__ == "__main__":
    main()