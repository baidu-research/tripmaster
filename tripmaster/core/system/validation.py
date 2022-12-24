"""
validations for the system
"""
import enum
import io
from typing import Type

import networkx as nx

from tripmaster.core.components.contract import TMContractGraph
from tripmaster.core.concepts.contract import TMContract, TMContractChannel
from tripmaster import logging
from tripmaster.utils.visualization import dot2image

logger = logging.getLogger(__name__)


class TMContractGraphNodeType(enum.Enum):
    Contract = enum.auto()
    Entry = enum.auto()
    Component = enum.auto()
    SubSystem = enum.auto()

class TMSystemValidator(object):


    def static_validate(self, system_type):


        task = system_type.TaskType
        tp_modeler = system_type.TPModelerType
        problem = system_type.ProblemType
        pm_modeler = system_type.PMModelerType
        machine = system_type.MachineType

        contract_graph = TMContractGraph()

        task.attach(contract_graph, role="Task")
        if tp_modeler:
            tp_modeler.attach(contract_graph, role="TPModeler")
        problem.attach(contract_graph, role="Problem")
        if pm_modeler:
            pm_modeler.attach(contract_graph, role="PMModeler")
        machine.attach(contract_graph, role="Machine")

        task_tpmodeler_contract = TMContract.parse(system_type.Task2ModelerContract)
        tpmodeler_problem_contract = TMContract.parse(system_type.Modeler2ProblemContract)
        task_problem_contract = TMContract.parse(system_type.Task2ProblemContract)
        problem_pmmodeler_contract = TMContract.parse(system_type.Problem2ModelerContract)
        pmmodeler_machine_contract = TMContract.parse(system_type.Modeler2MachineContract)
        problem_machine_contract = TMContract.parse(system_type.Problem2MachineContract)

        if tp_modeler is not None:
            contract_graph.connect_components(
                component1=task, role1="Task",
                component2=tp_modeler, role2="TPModeler",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: task_tpmodeler_contract,
                                  TMContractChannel.Target: task_tpmodeler_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(task_tpmodeler_contract)}
                )

            contract_graph.connect_components(
                component1=tp_modeler, role1="TPModeler",
                component2=problem, role2="Problem",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: tpmodeler_problem_contract,
                                  TMContractChannel.Target: tpmodeler_problem_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(tpmodeler_problem_contract)}
                )
        else:
            contract_graph.connect_components(
                component1=task, role1="Task",
                component2=problem, role2="Problem",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: task_problem_contract,
                                  TMContractChannel.Target: task_problem_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(task_problem_contract)}
            )


        if pm_modeler is not None:
            contract_graph.connect_components(
                component1=problem, role1="Problem",
                component2=pm_modeler, role2="PMModeler",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: problem_pmmodeler_contract,
                                  TMContractChannel.Target: problem_pmmodeler_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(problem_pmmodeler_contract)}
            )

            contract_graph.connect_components(
                component1=pm_modeler, role1="PMModeler",
                component2=machine, role2="Machine",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: pmmodeler_machine_contract,
                                  TMContractChannel.Target: pmmodeler_machine_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(pmmodeler_machine_contract)}
            )
        else:
            contract_graph.connect_components(
                component1=problem, role1="Problem",
                component2=machine, role2="Machine",
                forward_channel_mapping={TMContractChannel.Source: TMContractChannel.Source,
                                         TMContractChannel.Target: TMContractChannel.Target},
                backward_channel_mapping={TMContractChannel.Inference: TMContractChannel.Inference},
                forward_contract={TMContractChannel.Source: problem_machine_contract,
                                  TMContractChannel.Target: problem_machine_contract},
                backward_contract={TMContractChannel.Inference: TMContract.inverse(problem_machine_contract)}
            )

        contract_graph.visualize("contracts.svg")