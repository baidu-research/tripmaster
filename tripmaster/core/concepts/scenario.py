"""
TMScenario: Scenario class for Machine Learning
"""

import enum

class TMScenario(enum.Enum):
    """
    TMScenario
    """
    # Learning scenario. Typical Behavior:
    #   the data will have target field;
    #   the modeler can update itself;
    #   the machine forward will:
    #       return logits for calculating the loss;
    #       do not return the prediction
    Learning = enum.auto()

    # Evaluation scenario. Typical Behavior:
    #  the data will have target field;
    #  the modeler can not update itself;
    #  the machine forward will:
    #       return logits for calculating the loss;
    #       return the prediction for evaluation
    Evaluation = enum.auto()

    # Inference scenario. Typical Behavior:
    #  the data will not have target field;
    #  the modeler can not update itself;
    #  the machine forward will:
    #       do not return logits for calculating the loss;
    #       return the prediction results
    Inference = enum.auto()