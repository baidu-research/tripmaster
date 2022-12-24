"""
TM schema
"""
import collections
import copy

from schema import Schema


class TMSchema(object):
    """
    TMSchema
    """

    def __init__(self, schema_data, error=None, name=None, description=None, as_reference=False):

        from schema import _priority, DICT

        if isinstance(schema_data, Schema):
            schema_data = copy.copy(schema_data._schema)
        elif isinstance(schema_data, TMSchema):
            schema_data = copy.copy(schema_data.data())

        assert _priority(schema_data) == DICT, schema_data

        self.schema = Schema(schema_data, error=error, ignore_extra_keys=True, name=name,
                         description=description, as_reference=as_reference)

    def data(self):

        return self.schema._schema

    def validate(self, data):
        if not self.schema._schema:
            return dict((key, value) for key, value in data.items() if "@" not in key)
        else:
            try:
                return self.schema.validate(data)
            except Exception as e:
                ic(self.data())
                ic(data.keys())
                raise

    def __getattr__(self, item):
        """

        Args:
            item:

        Returns:

        """
        if item.startswith("__"):  # this allows for deepcopy
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(item)
            )
        return getattr(self.schema, item)

    def add(self, schema_to_add):
        """

        Args:
            schema_to_add:

        Returns:

        """
        from schema import _priority, DICT

        assert _priority(schema_to_add) == DICT

        self.schema._schema.update(schema_to_add)

    def entries(self):

        return set(self.schema._schema.keys())

    def nested_entries(self):

        key_list = []

        def __get_keys(dict_data, prefix, result_list):
            for key, value in dict_data.items():

                result_list.append(prefix + key)
                if isinstance(value, dict):
                    this_prefix = prefix + key + "."
                    __get_keys(value, this_prefix, result_list)
                elif isinstance(value, (list, tuple)) and isinstance(value[0], dict):
                    this_prefix = prefix + key + ":"
                    __get_keys(value[0], this_prefix, result_list)

        __get_keys(self.schema._schema, "", key_list)

        return key_list

    def is_valid(self, data):

        if self.schema.is_valid(data):
            return True
        else:
            ic(self.data())
            ic(data)
            return False


class TMChannelSchema(collections.UserDict):
    """
    TMChannelSchema
    """

    def __init__(self, channel_schema_data):
        super().__init__()
        channel_schema = dict((key, TMSchema(value)) for key, value in channel_schema_data.items())
        self.update(channel_schema)

        from collections import ChainMap
        schema_data = dict()
        for key, schema in channel_schema.items():
            schema_data.update(schema.data())
        self.all_schema = TMSchema(schema_data)

    def all(self):
        return self.all_schema

