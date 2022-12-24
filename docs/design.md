# Design Principles of the TripMaster Framework

## Destinations

The TripMaster framework is designed for three significant destinations: Reusability, Reproducibility, and Easy to Use.

### Reusability 

People are building tons of machine learning systems each day. If these systems are built with reusable components, 
a bunch of time could be saved. An ideal approach to building a system is like the 'Simulink' system of Matlab: People drag 
the component to the diagram and link them with control flows, and then a system is finished. 

TripMaster framework follows the same methodology. TripMaster framework uses a dictionary to express each sample, with keys representing 
data fields. Each `component` in the TripMaster framework is a fully functional, self-contained, and self-declared unit, 
where self-declaration means that it should declare what data it needs 
and what data it will produce using dictionary schema, where each key represents a requirement or provision. The components under 
the TripMaster framework can be connected by `contract` to form a system, where the `contract` is responsible for mapping the 
provisions of the former component to the requirements of the later component by mapping data fields. If more complex operations
are needed to bridge the provided data of the former component and the required data of the later component, one can write their own 
`modeler` component (also reusable) as the bridge. The design of `component`, `contract`, `modeler` provides the basic 
level reusability to the TripMaster framework. 

Based on this basic level of reusability support, the TripMaster framework designs the architecture of a machine learning system containing 
following components:

* Task: defining the task to fulfill, sub-components: Evaluator (optional)
* Problem: defining the mathematical problem of the task, sub-components: Evaluator (optional)
* Machine: defining the solver of the problem, sub-components: Loss, Evaluator (optional):
* Operator: running the machine. It has two subtypes:
  * Learner: train the machine from data;
  * Inferencer: inference from the data and output the results;
* System: containing all above components and driving them to accomplish the entire machine learning procedure;
* Application: applying the system on a special dataset, obtaining machine or the inference results; 
* Pipeline: chaining a sequence of inference applications, frequently used in the NLP domain. 

The philosophy behind this design is that one can build many different mathematical models for the same task and can also use 
the same mathematical model to model various tasks. So the problem components can be reused. Again, one can build many machines 
to solve one problem and use one machine to solve many problems (with the help of `contract` and `modeler`). 
This design builds the high-level reusability of the TripMaster framework. 

### Reproducibility

The difficulties in publishing reproducible research are that even the code of the paper is published, readers are 
generally found that it is usually difficult to understand, not easy to reconfigure to run in your environment, 
and not easy to use their algorithm in your system. 

The reasons are as follows:
* Most machine learning codes focus on machine building and optimization and pay less attention to the data pipeline and data schema. The data flow code is always written arbitrarily, and the procedure is challenging to understand.   
* The configurations of the whole system scatter everywhere in the code;
* The components are not self-contained and depend on each other, including shared variables, codes across modules, etc. 

TripMaster framework solves these problems by:
* TripMaster encloses the entire lifetime of data in the system: data is first input as the task-level raw data, then converted to math-level data, and at last be converted machine-level batched data. In each phase, the data schema is clearly defined and easy to understand, which significantly alleviates the burden of understanding the data flow;
* TripMaster requires each component to be configurable and implement a centered configuration strategy to allow the user to configure the system in one YAML file;
* TripMaster's components are self-contained, and its interface is clearly declared by its schema declaration. One can just import that module and feed the correct data, then the components will work for you.  

### Easy to Use  

TripMaster provides the following mechanism for less code to implement full functionality:

* Plugin mechanism for strategies. TripMaster provides a lot of predefined strategies. Users can use them by setting the type of themin the slot or configuring them through configuration file;
* Unified serialization mechanism. Users can serialize every component together with their hyper-parameters, including the entire system;
* A visualization tool to visualize the data flow of an application;


## Target User 

With the above design principles, the target users of the TripMaster framework is clear:

* It is suitable for users who want to maintain a long-term or median/large-scale repo that contains many machine learning algorithms. TripMaster will help them increase the reusability and the organization of the code repo.  
* It is suitable for users who want to publish reproducible research results. TripMaster will help them improve the reproducibility and thus the potential influence of their code and research. 
* For users who only want to write codes that run a few times and then abandon them, TripMaster may still be helpful for easy and comfortable coding through the functions for the purpose of `easy to use`.


However, TripMaster cannot guarantee your code satisfies the above design principles. You can also write very bad non-reusable or non-reproducible 
code. TripMaster helps to organize your idea and your code. You still need to optimize your concept and code inside the TripMaster
framework. 
