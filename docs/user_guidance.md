

# User Guidance 

## Project Organization 

We recommend creating a package for each app, using the following manner to organize your project:

```text
App                                    # the app package 
|-- example_config.yaml                # example config for better understanding 
|-- inference_app.py                   # inference app definition
|-- learning_app.py                    # learning app definition 
|-- system.py                          # system definition
|-- natural                            # 'natural' means task level code, including task definitions, 
                                         task structures, task evaluators, and task2problem modelers. 
                                         We show these components in packages. They can also be modules (.py files). 
|   |-- evaluators                          # task evaluator 
|   |-- modeler                             # task to problem modeler
|   |   |-- dataprocessor                       # we recommend to develop reusable data processor 
                                                  for building the modeler 
|   |-- resource                            # external resources 
|   |-- structure                           # structures for implementing the task data 
|   |-- task                                # task definition 
|   |-- utils                               # utilities 
|-- rational                           # 'rational' means problem/machine level code, including definitions, 
                                         structures, evaluators, problem2machine modelers.
                                         We show these components in packages. They can also be modules (.py files). 
|   |-- evaluators                          # problem/machine evaluators 
|   |   |-- metrics                             # we recommend to develop reusable metrics 
                                                  for building the evaluator
|   |-- loss                                # loss definition 
|   |-- machine                             # machine definition 
|   |-- modeler                             # problem/machine evaluators 
|   |-- operator                            # operator definition
|   |-- problem                             # problem definition
|   |-- structure                           # structures for implementing the problem/machine data
```

## Re-usability

We highly recommend user to implement their app using re-usable components. 
You may notice that the `tripmaster` project is also organized using above structure. So if you implement a component and 
you think it may be helpful for others, you can easily find the correct folder to place the module and submit a 
`pull request` to `tripmaster` project. 

## The bridge between Data Phase and Operation Phase 

When building the machine and the operator, we may need some statistics of the data, including the size of vocabulary, 
the number of samples, etc. One can override the `on_data_phase_finished` method to obtain the information to print it or 
set the hyper_params of the machines and operators. 

We recommend following procedure in training procedure:

1. Use data_mode to generate the machine level data, and save the system without the machine and operator, to save the
states of task, problem, and modelers.
2. Set data_mode to False, load the saved system and the machine-level data for training. 
3. You can choose to training from scratch or continuing with existing results by setting the job.operation to `from_scratch`
or `continue`.

Through this way:

1. You can debug and train your machine efficiently with machine-level data, without to build data from scratch each time;
2. The status of task, problem and modeler are honestly saved, so even you use the macine-level data, you can still evaluate
the performance in problem-level and task-level.

