import enum
import io
from typing import Type

import networkx as nx
from tripmaster.core.concepts.schema import TMSchema

from tripmaster.core.concepts.contract import TMContract, TMContracted


class TMContractGraphNodeType(enum.Enum):
    Contract = enum.auto()
    Entry = enum.auto()
    Component = enum.auto()
    SubSystem = enum.auto()


class TMContractGraph(nx.DiGraph):

    def __init__(self):
        super().__init__()

        self.contract_id = 0

    def add_entry(self, entry_id, schema):
        """

        Args:
            entry_id:
            schema:

        Returns:

        """
        self.add_node(entry_id, schema=schema, type=TMContractGraphNodeType.Entry)

    def add_contract(self, entry1, entry2, contract):
        """

        Args:
            entry1:
            entry2:
            contract:

        Returns:

        """

        contract_node_id = f"contract_{self.contract_id}"

        s_provide = self.nodes[entry1]["schema"]
        e_require = self.nodes[entry2]["schema"]
        assert isinstance(s_provide, TMSchema)
        assert isinstance(e_require, TMSchema)

        if contract is not None:
            if isinstance(contract, TMContract):
                contract = contract.map

            contract_adapted = [contract[x] for x in s_provide.entries() if x in contract]

            if not set(e_require.entries()) <= set(contract_adapted):
                error = True
            else:
                error = False
                contract = dict((k, v) for k, v in contract.items() if v in e_require.entries())
        else:

            if not set(s_provide.entries()) >= set(e_require.entries()):
                error = True
            else:
                error = False

        self.add_node(contract_node_id, type=TMContractGraphNodeType.Contract,
                      contract=contract, error=error)

        self.add_edge(entry1, contract_node_id)
        self.add_edge(contract_node_id, entry2)

        self.contract_id += 1

    def add_component(self, component_id, request_entries, provision_entries):
        """

        Args:
            component_id:
            entries:

        Returns:

        """
        self.add_node(component_id, type=TMContractGraphNodeType.Component)

        for entry in request_entries:
            self.add_edge(component_id, entry)

        for entry in provision_entries:
            self.add_edge(component_id, entry)

    def connect_components(self, component1: Type[TMContracted], role1,
                           component2: Type[TMContracted], role2,
                           forward_channel_mapping, backward_channel_mapping,
                           forward_contract, backward_contract):

        for channel1, channel2 in forward_channel_mapping.items():
            entry1 = component1.make_contract_entry(role1, forward=True, request=False, channel=channel1)
            entry2 = component2.make_contract_entry(role2, forward=True, request=True, channel=channel2)

            self.add_contract(entry1, entry2, forward_contract[channel2])

        for channel1, channel2 in backward_channel_mapping.items():
            entry1 = component1.make_contract_entry(role1, forward=False, request=True, channel=channel1)
            entry2 = component2.make_contract_entry(role2, forward=False, request=False, channel=channel2)

            self.add_contract(entry2, entry1, backward_contract[channel2])


    def connect_consumer(self, component: Type[TMContracted], component_role,
                         consumer: Type[TMContracted], consumer_role,
                         channel_mapping, contract):

        for channel, consumer_channel in channel_mapping.items():
            if channel in component.ForwardRequestSchema:
                entry1 = component.make_contract_entry(component_role, forward=True, request=True, channel=channel)
            elif channel in component.BackwardRequestSchema:
                entry1 = component.make_contract_entry(component_role, forward=False, request=True, channel=channel)
            elif channel in component.ForwardProvisionSchema:
                entry1 = component.make_contract_entry(component_role, forward=True, request=False, channel=channel)
            elif channel in component.BackwardProvisionSchema:
                entry1 = component.make_contract_entry(component_role, forward=False, request=False, channel=channel)
            else:
                raise Exception(f"the {component} does not support channel {channel}")

            entry2 = consumer.make_contract_entry(consumer_role, forward=True, request=True,
                                                      channel=consumer_channel)

            self.add_contract(entry1, entry2, contract[consumer_channel])



    def add_subsystem(self, system_id, components):

        self.add_node(system_id, type=TMContractGraphNodeType.SubSystem)

        for component in components:
            self.add_edge(system_id, component)

    def validate(self):

        error = False
        for n in self.nodes():
            if self.nodes[n]["type"] == TMContractGraphNodeType.Contract \
                    and self.nodes[n]["error"]:
                error = True
                break

        return error

    def finish(self):

        for contract in self.nodes():
            # print(f"node = {(contract, self.nodes[contract])}")
            # print(f"predcessor = {[(x, self.nodes[x]) for x in self.predecessors(contract)]}")
            # print(f"successor = {[(x, self.nodes[x]) for x in self.successors(contract)]}")
            if self.nodes[contract]["type"] != TMContractGraphNodeType.Contract:
                continue

            provider = list(self.predecessors(contract))[0]
            requirer = list(self.successors(contract))[0]

            component1 = [x for x in self.predecessors(provider)
                          if self.nodes[x]["type"] == TMContractGraphNodeType.Component][0]
            subsystem1 = [x for x in self.predecessors(component1)
                          if self.nodes[x]["type"] == TMContractGraphNodeType.SubSystem]
            component2 = [x for x in self.predecessors(requirer)
                          if self.nodes[x]["type"] == TMContractGraphNodeType.Component][0]
            subsystem2 = [x for x in self.predecessors(component2)
                          if self.nodes[x]["type"] == TMContractGraphNodeType.SubSystem]

            assert len(subsystem1) <= 1 and len(subsystem2) <= 1

            if len(subsystem1) == len(subsystem2) == 1 and subsystem1[0] == subsystem2[0]:
                subsystem = list(subsystem1)[0]

                self.add_edge(subsystem, contract)

    def visualize_entry(self, entry):

        try:
            schema = self.nodes[entry]["schema"]
        except:
            schema = {}
        import io

        dot_string = io.StringIO()
        dot_string.write('<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"> \n')
        dot_string.write(f"<TR> <TD> {entry} </TD>  </TR> \n")
        for k in schema.entries():
            dot_string.write(f"<TR> <TD> {k} </TD>  </TR>")
        dot_string.write("</TABLE>")
        return f'"{entry}"\t[label=<{dot_string.getvalue()}>]; \n'

    def visualize_contract(self, contract_id):

        import io

        contract = self.nodes[contract_id]["contract"]
        error = self.nodes[contract_id]["error"]

        color = "red" if error else "black"

        provider = list(self.predecessors(contract_id))[0]
        requirer = list(self.successors(contract_id))[0]

        if contract:
            dot_string = io.StringIO()
            dot_string.write('<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"> \n')
            for k, v in contract.items():
                dot_string.write(f"<TR> <TD> {k} </TD> <TD> {v} </TD> </TR>")
            dot_string.write("</TABLE>")

            return f'{contract_id}\t[label=<{dot_string.getvalue()}>, color={color}]; \n' \
                   f'\"{provider}\"\t->\t\"{contract_id}\" [color={color}];\n' \
                   f'\"{contract_id}\"\t->\t\"{requirer}\" [color={color}];\n'
        else:
            return f'\"{provider}\"\t->\t\"{requirer}\" [color={color}];\n'

    def visualize_component(self, component):
        """

        Args:
            component:

        Returns:

        """

        dot_string = io.StringIO()

        dot_string.write(f'subgraph "cluster_{component}" {{label="{component}" color=blue; \n')

        vis_node_label = f"\"{component}\"\t[label={component}]; \n"
        #        dot_string.write(vis_node_label)

        require_entries = list([v for v in self.predecessors(component)
                                if self.nodes[v]["type"] == TMContractGraphNodeType.Entry])
        provide_entries = list(self.successors(component))

        for entry in require_entries + provide_entries:
            dot_string.write(self.visualize_entry(entry))

        #        for entry in require_entries:
        #            dot_string.write('\"{0}\"\t->\t\"{1}\";\n'.format(
        #                entry, component
        #            ))
        #        for entry in provide_entries:
        #            dot_string.write('\"{0}\"\t->\t\"{1}\";\n'.format(
        #                component, entry
        #            ))

        dot_string.write("\n }; \n")

        return dot_string.getvalue()

    def visualize_subsystem(self, subsystem):

        dot_string = io.StringIO()

        dot_string.write(f"subgraph cluster_{subsystem} {{label={subsystem} \n")

        for child in self.successors(subsystem):
            if self.nodes[child]["type"] == TMContractGraphNodeType.Component:
                dot_string.write(self.visualize_component(child))
            elif self.nodes[child]["type"] == TMContractGraphNodeType.Contract:
                dot_string.write(self.visualize_contract(child))
            else:
                raise Exception(f"unknown node type: {self.nodes[child]['type']}")

        dot_string.write("\n }; \n")

        return dot_string.getvalue()

    def visualize(self, output_file_path, format=None):

        dot_string = io.StringIO()

        dot_string.write("strict digraph {\n")
        dot_string.write("node [shape=box];\n")

        subsystems = [node_id for node_id in self.nodes
                      if self.nodes[node_id]["type"] == TMContractGraphNodeType.SubSystem]

        processed_nodes = set()
        for subsystem in subsystems:
            dot_string.write(self.visualize_subsystem(subsystem))
            processed_nodes.update(self.successors(subsystem))

        not_processed_components = [node_id for node_id in self.nodes
                                    if self.nodes[node_id]["type"] == TMContractGraphNodeType.Component and
                                    node_id not in processed_nodes]
        not_processed_contracts = [node_id for node_id in self.nodes
                                   if self.nodes[node_id]["type"] == TMContractGraphNodeType.Contract and
                                   node_id not in processed_nodes]

        for component in not_processed_components:
            dot_string.write(self.visualize_component(component))

        for contract in not_processed_contracts:
            dot_string.write(self.visualize_contract(contract))

        #        for s, e in self.edges():

        #            dot_string.write('\"{0}\"\t->\t\"{1}\";\n'.format(s, e))

        dot_string.write("}\n")

        dot_string = dot_string.getvalue()

        with open(output_file_path + ".dot", "w") as f:
            f.write(dot_string)

        from tripmaster.utils.visualization import dot2image

        image = dot2image(dot_string, output_file_path, format=format)